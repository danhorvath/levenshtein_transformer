import torch
from transformer.modules import subsequent_mask


def greedy_decode(model, src, src_mask, max_len, start_symbol, stop_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys.detach().clone(),
                           subsequent_mask(ys.size(1)).detach().clone().type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # early stop
        if next_word == stop_symbol:
            break
    return ys
