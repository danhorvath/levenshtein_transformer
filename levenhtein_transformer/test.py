import numpy as np
import torch
from levenhtein_transformer.data import BatchWithNoise


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = torch.tensor(data, requires_grad=False)
        tgt = torch.tensor([] + data + [], requires_grad=False)
        yield BatchWithNoise(src, tgt, 0)
