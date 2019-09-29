from datetime import datetime
import torch
import torch.nn as nn
from torchtext import data, datasets


from functions import greedy_decode
from train import LabelSmoothing, NoamOpt, run_epoch
from multi_gpu_loss_compute import MultiGPULossCompute
from models import Transformer
from data import batch_size_fn, MyIterator, rebatch
from validator import validate


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
NUM_LAYERS = 6
BATCH_SIZE = 25000
MAX_LEN = 100
MIN_FREQ = 2
FACTOR = 2
WARMUP = 4000
LR = 0
BETAS = [0.9, 0.98]
EPSILON = 1e-9
MAX_STEPS = 1e5

devices = list(range(torch.cuda.device_count()))
print('Selected devices: ', devices)


def tokenize_bpe(text): return text.split()


SRC = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_bpe, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)

train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                         train='newstest2014.tok.bpe.32000',
                                         validation='newstest2013.tok.bpe.32000',
                                         test='newstest2014.tok.bpe.32000',
                                         fields=(SRC, TGT),
                                         root='./.data/')

print('Train set length: ', len(train))

# building shared vocabulary
TGT.build_vocab(train.src, train.trg, min_freq=1)
SRC.vocab = TGT.vocab

print('Source vocab length: ', len(SRC.vocab.itos))
print('Target vocab length: ', len(TGT.vocab.itos))

pad_idx = TGT.vocab.stoi[BLANK_WORD]

test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=torch.device(0), 
    repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)


print('Loading model...')
model = Transformer(len(SRC.vocab), len(TGT.vocab), N=NUM_LAYERS)
# model.load_state_dict(torch.load('./en-de__Sep-23-2019_18-31.pt'))
model.cuda()
print('Model loaded.')

test_bleu = validate(model, test_iter, SRC, TGT,
                     BOS_WORD, EOS_WORD, BLANK_WORD, MAX_LEN, logging=True)
print('Test Bleu score: %f' % (test_bleu))


# tokens = list()
# with open('.data/wmt14/vocab.bpe.32000') as f:
#     for line in f:
#         tokens.append(line.rstrip('\n'))

# tokens.append(BOS_WORD)
# tokens.append(EOS_WORD)
# tokens.append(BLANK_WORD)
