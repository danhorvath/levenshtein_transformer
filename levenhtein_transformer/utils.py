import torch
from torch import Tensor
import levenhtein_transformer.libnat as libnat


# based on fairseq.libnat


def _get_ins_targets(pred: Tensor, target: Tensor, padding_idx: int, unk_idx: int) -> \
        (Tensor, Tensor, Tensor):
    """
    :param pred: Tensor
    :param target: Tensor
    :param padding_idx: long
    :param unk_idx: long
    :return: word_pred_input, word_pred_tgt_masks, ins_targets
    """
    in_seq_len = pred.size(1)
    out_seq_len = target.size(1)

    with torch.cuda.device_of(pred):
        # removing padding
        pred_list = [[t for t in s if t != padding_idx] for i, s in enumerate(pred.tolist())]
        target_list = [[t for t in s if t != padding_idx] for i, s in enumerate(target.tolist())]

        full_labels = libnat.suggested_ed2_path(pred_list, target_list, padding_idx)

        # get insertion target with number of insertions eg. [0, 2, 1, 0, 2, 0]
        insertion_tgts = [[len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels]

        # generate labels
        word_pred_tgt_masks = []
        for insertion_tgt in insertion_tgts:
            word_gen_mask = []
            # get mask for word generation, eg: [0, 1, 1, 0, 1, 0, 0, 1, 1]
            for beam_size in insertion_tgt[1:-1]:  # HACK 1:-1
                word_gen_mask += [0] + [1 for _ in range(beam_size)]

            # add padding
            word_pred_tgt_masks.append(word_gen_mask + [0 for _ in range(out_seq_len - len(word_gen_mask))])

        ins_targets = [
            insertion_tgt[1:-1] +
            [0 for _ in range(in_seq_len - 1 - len(insertion_tgt[1:-1]))]
            for insertion_tgt in insertion_tgts
        ]

        # transform to tensor

        # word_pred_tgt_masks = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, ..., 0]
        word_pred_tgt_masks = torch.tensor(word_pred_tgt_masks, device=target.device).bool()

        # ins_targets = [0, 2, 1, 0, 2, 0, 0, 0, ..., 0]
        ins_targets = torch.tensor(ins_targets, device=pred.device)

        # word_pred_tgt = [0, <unk>, <unk>, 0, <unk>, 0, 0, <unk>, <unk>, 0, 0, ..., 0]
        word_pred_input = target.masked_fill(word_pred_tgt_masks, unk_idx)
    return word_pred_input, word_pred_tgt_masks, ins_targets


def _get_del_targets(prediction, target, padding_idx):
    out_seq_len = target.size(1)

    with torch.cuda.device_of(prediction):
        prediction_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(prediction.tolist())
        ]
        target_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(target.tolist())
        ]

        # get labels in form of [insert1, insert2, ..., insertn, [del1, del2, ..., deln]]
        full_labels = libnat.suggested_ed2_path(
            prediction_list, target_list, padding_idx
        )

        word_del_targets = [b[-1] for b in full_labels]
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]

        # transform to tensor
        word_del_targets = torch.tensor(word_del_targets, device=target.device)
    return word_del_targets


def _get_del_ins_targets(in_tokens, out_tokens, padding_idx):
    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

        full_labels = libnat.suggested_ed2_path(in_tokens_list, out_tokens_list, padding_idx)

        # deletion target, eg: [0, 0, 1, 0, 1, 0]
        word_del_targets = [b[-1] for b in full_labels]
        # add padding
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]

        # insertion target with number of words to be inserted, eg [0, 0, 3, 0, 2]
        mask_inputs = [
            [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
        ]
        mask_ins_targets = [
            mask_input[1:-1] +
            [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
            for mask_input in mask_inputs
        ]

        # transform to tensor
        mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
        word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
    return word_del_targets, mask_ins_targets


def _apply_ins_masks(in_tokens: Tensor, mask_ins_pred: Tensor, pad: int, unk: int, eos: int):
    in_masks = in_tokens.ne(pad)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
            torch.arange(out_max_len, device=out_lengths.device)[None, :]
            < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
            .fill_(pad)
            .masked_fill_(out_masks, unk)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    return out_tokens


def _apply_ins_words(in_tokens: Tensor, word_ins_pred: Tensor, unk: int):
    word_ins_masks = in_tokens.eq(unk)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    return out_tokens


def _apply_del_words(in_tokens: Tensor, word_del_pred: Tensor, pad: int, bos: int,
                     eos: int) -> Tensor:
    # apply deletion to a tensor
    in_masks = in_tokens.ne(pad)
    bos_eos_masks = in_tokens.eq(bos) | in_tokens.eq(eos)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = (
        torch.arange(max_len, device=in_tokens.device)[None, :]
            .expand_as(in_tokens)
            .contiguous()
            .masked_fill_(word_del_pred, max_len)
            .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(
        word_del_pred, pad).gather(1, reordering)

    return out_tokens


# from fairseq model_utils


def skip_tensors(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [skip_tensors(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: skip_tensors(v, mask) for k, v in x.items()}

    raise NotImplementedError


def fill_tensors(x: Tensor, mask: Tensor, y: Tensor, padding_idx: int) -> Tensor:
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


def inject_noise(target_tokens: Tensor, pad, bos, eos):
    with torch.cuda.device_of(target_tokens):
        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
        target_score.masked_fill_(target_mask, 1)

        # reorder the numbers randomly, with bos and eos at the beginning and paddings at the end
        # ['<bos>', 'asd', 'kek', 'lol', '<bos>', '<pad>', '<pad>', '<pad>'] =>
        # ['<bos>', '<bos>',  'kek', 'lol', 'asd','<pad>', '<pad>', '<pad>']
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(1, keepdim=True)

        # do not delete <bos> and <eos> (we assign 0 score for them)
        # assign a new random length for each line, where: 2 < new_length < original_length
        target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(target_score.size(0), 1).uniform_()).long()
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        # remove tokens after the cutoff
        prev_target_tokens = target_tokens.gather(1, target_rank).masked_fill_(target_cutoff, pad) \
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])

        # remove unnecessary paddings
        prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.ne(pad).sum(1).max()]

    return prev_target_tokens


def initialize_output_tokens(src_tokens: Tensor, bos: int, eos: int):
    initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
    initial_output_tokens[:, 0] = bos
    initial_output_tokens[:, 1] = eos

    return initial_output_tokens


def pad_tensors_in_dim(x_org: Tensor, y_org: Tensor, dim: int, pad: int) -> (Tensor, Tensor):
    x = x_org.detach().clone()
    y = y_org.detach().clone()
    x_shape = [*x.size()]
    y_shape = [*y.size()]

    if y_shape[dim] == x_shape[dim]:
        return x, y
    elif y_shape[dim] > x_shape[dim]:
        pad_shape = x_shape
        pad_shape[dim] = y_shape[dim] - x_shape[dim]
        padding_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device).fill_(pad)
        padded_x = torch.cat([x, padding_tensor], dim=dim)
        return padded_x, y
    elif y_shape[dim] < x_shape[dim]:
        pad_shape = y_shape
        pad_shape[dim] = x_shape[dim] - y_shape[dim]
        padding_tensor = torch.zeros(pad_shape, dtype=y.dtype, device=y.device).fill_(pad)
        padded_y = torch.cat([y, padding_tensor], dim=dim)
        return x, padded_y


def pad_tensor_to_length(x_org: Tensor, len: int, dim: int, pad: int) -> Tensor:
    x = x_org.detach().clone()
    x_shape = [*x.size()]

    if x_shape[dim] <= len:
        return x
    else:
        pad_shape = x_shape
        pad_shape[dim] = len - x_shape[dim]
        padding_tensor = torch.zeros(pad_shape, dtype=x.dtype, device=x.device).fill_(pad)
        padded_x = torch.cat([x, padding_tensor], dim=dim)
        return padded_x
