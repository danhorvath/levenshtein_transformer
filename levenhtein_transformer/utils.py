import torch
import levenhtein_transformer.libnat as libnat


# based on fairseq.libnat


def _get_ins_targets(pred, target, padding_idx, unk_idx):
    """
    :param pred: torch.Tensor
    :param target: torch.Tensor
    :param padding_idx: long
    :param unk_idx: long
    :return: word_pred_tgt, word_pred_tgt_masks, ins_targets
    """
    in_seq_len = pred.size(1)
    out_seq_len = target.size(1)

    with torch.cuda.device_of(pred):
        # removing padding
        pred_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(pred.tolist())
        ]
        target_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(target.tolist())
        ]

    full_labels = libnat.suggested_ed2_path(
        pred_list, target_list, padding_idx
    )

    # get insertion target with number of insertions eg. [0, 0, 4, 0]
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
    word_pred_tgt_masks = torch.tensor(word_pred_tgt_masks, device=target.device).bool()
    ins_targets = torch.tensor(ins_targets, device=pred.device)
    word_pred_tgt = target.masked_fill(word_pred_tgt_masks, unk_idx)
    return word_pred_tgt, word_pred_tgt_masks, ins_targets


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

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )

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


def _apply_ins_masks(in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx):
    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
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
            .fill_(padding_idx)
            .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(
        word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

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
        word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(
            word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.).gather(1, _reordering)

    return out_tokens, out_scores, out_attn


# from fairseq model_utils


def skip_tensors(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [skip_tensors(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: skip_tensors(v, mask) for k, v in x.items()}

    raise NotImplementedError


def fill_tensors(x, mask, y, padding_idx):
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
