import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

from data import data_gen, SimpleLossCompute, Batch
from functions import greedy_decode
from models import Transformer
from train import LabelSmoothing, NoamOpt, run_epoch

if __name__ == '__main__':
    # Greedy decoding

    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = Transformer(V, V, N=2)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(data_gen(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None)))

    model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

    # batch = 2
    # dim = 6
    # a = torch.ones(batch, dim)
    # print(a)
    # print((a != 0).unsqueeze((-2)))

    # crit = LabelSmoothing(5, 0, 0.4)
    # predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
    #                              [0, 0.2, 0.7, 0.1, 0],
    #                              [0, 0.2, 0.7, 0.1, 0]])
    # v = crit(Variable(predict.log()),
    #          Variable(torch.LongTensor([2, 1, 0])))
    #
    # print(torch.LongTensor([2, 1, 0]))
    # print(crit.true_dist)
