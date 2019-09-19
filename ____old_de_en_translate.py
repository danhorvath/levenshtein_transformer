from datetime import datetime
import spacy
import torch
import torch.nn as nn
from torchtext import data, datasets


from functions import greedy_decode
from train import LabelSmoothing, NoamOpt, run_epoch
from multi_gpu_loss_compute import MultiGPULossCompute
from models import Transformer
from data import batch_size_fn, MyIterator, rebatch


import wandb
wandb.init(project="transformer")

# GPUs to use
devices = list(range(torch.cuda.device_count()))

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD)
NUM_LAYERS = 6
BATCH_SIZE = 12000

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

MAX_LEN = 100
train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                         filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

pad_idx = TGT.vocab.stoi[BLANK_WORD]
model = Transformer(len(SRC.vocab), len(TGT.vocab), N=NUM_LAYERS)

MODEL_SIZE = model.src_embed[0].d_model
FACTOR = 1
WARMUP = 2000
LR = 0
BETAS = [0.9, 0.98]
EPSILON = 1e-9

wandb.config.update({'epochs': 10,
                     'src_vocab_length': len(SRC.vocab),
                     'target_vocab_length': len(TGT.vocab),
                     'num_layers': NUM_LAYERS,
                     'batch_size': BATCH_SIZE,
                     'model_size': MODEL_SIZE,
                     'factor': FACTOR,
                     'warmmup': WARMUP,
                     'learning_rate': LR,
                     'epsilon': EPSILON
                     })

model.cuda()
criterion = LabelSmoothing(
    size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
criterion.cuda()

train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device(0), repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device(0), repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)
model_par = nn.DataParallel(model, device_ids=devices)

model_opt = NoamOpt(MODEL_SIZE, FACTOR, WARMUP,
                    torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=EPSILON))

wandb.watch(model)

for epoch in range(10):
    model_par.train()
    run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
              MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt), epoch)

    model_par.eval()
    loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                     MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None), epoch)
    print(loss)

current_date = datetime.now().strftime("%b-%d-%Y_%H-%M")
torch.save(model.state_dict(), 'de-en__'+current_date+'.pt')

for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, max_len=60,
                        start_symbol=TGT.vocab.stoi[BOS_WORD])
    print('Translation:', end='\t')
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == EOS_WORD:
            break
        print(sym, end=' ')
    print()
    print('Target:', end='\t')
    for i in range(batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == EOS_WORD:
            break
        print(sym, end=' ')
    print()
    break
