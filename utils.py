from datetime import datetime
import torch
import copy
import collections
import dill


def load_models(example_model, paths):
    models = []
    for path in paths:
        model = copy.deepcopy(example_model)
        model.load_state_dict(torch.load(path))
        models.append(copy.deepcopy(model))
    return models


def average(model, models):
    "Average models into model"
    for ps in zip(*[m.parameters() for m in [model] + models]):
        ps[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


def save_model(model, optimizer, loss, src_field, tgt_field, updates, epoch, prefix=''):
    if prefix != '':
        prefix += '_'
    current_date = datetime.now().strftime("%b-%d-%Y_%H-%M")
    file_name = prefix + 'en-de__' + current_date + '.pt'
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'loss': loss,
        'updates': updates
    }, file_name)
    torch.save(src_field, f'SRC_{len(src_field.vocab.itos)}.pt', pickle_module=dill)
    torch.save(tgt_field, f'TGT_{len(tgt_field.vocab.itos)}.pt', pickle_module=dill)
    print('Model is saved as {file_name}')


def bpe_to_words(sentence):
    new_sentence = []
    for i in range(len(sentence)):
        word = sentence[i]
        if word[-2:] == '@@' and i != len(sentence) - 1:
            sentence[i + 1] = word[:-2] + sentence[i + 1]
        else:
            new_sentence.append(word)
    return new_sentence


def vector_to_sentence(vector: torch.Tensor, field, eos_word: str, start_from=1):
    sentence = []
    for l in range(start_from, vector.size(0)):
        word = field.vocab.itos[vector[l]]
        if word == eos_word:
            break
        sentence.append(word)

    # fixing encoding
    sentence = ' '.join(bpe_to_words(sentence)).encode('utf-8').decode('latin-1')
    return sentence


def get_src_mask(src, BLANK_WORD_IDX):
    return (src != BLANK_WORD_IDX).unsqueeze(-2)


def average_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.
    Args:
      inputs: An iterable of string paths of checkpoints to load from.
    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter
            else:
                params_dict[k] += p

    averaged_params = collections.OrderedDict()
    for k, v in params_dict.items():
        averaged_params[k] = v
        averaged_params[k].div_(num_models)
    return averaged_params
