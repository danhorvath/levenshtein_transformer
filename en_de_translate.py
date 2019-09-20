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
from validator import validate


import wandb
wandb.init(project="transformer")


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


def main():
    devices = list(range(torch.cuda.device_count()))
    print('Selected devices: ', devices)

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    def tokenize_bpe(text):
        return text.split()

    SRC = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_bpe, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                             train='train.tok.clean.bpe.32000',
                                             validation='newstest2013.tok.bpe.32000',
                                             test='newstest2014.tok.bpe.32000',
                                             fields=(SRC, TGT),
                                             root='./.data/')
    print('Train set length: ', len(train))

    # building shared vocabulary
    SRC.build_vocab(train.src, train.trg, min_freq=MIN_FREQ)
    TGT.vocab = SRC.vocab

    print('Source vocab length: ', len(SRC.vocab.itos))
    print('Target vocab length: ', len(TGT.vocab.itos))

    pad_idx = TGT.vocab.stoi[BLANK_WORD]

    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    test_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device(0), repeat=False,
                           sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    model = Transformer(len(SRC.vocab), len(TGT.vocab), N=NUM_LAYERS)

    # weight tying
    model.src_embed[0].lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.generator.lookup_table.weight = model.tgt_embed[0].lookup_table.weight

    model.cuda()

    model_size = model.src_embed[0].d_model
    print('Model created with size of', model_size)

    wandb.config.update({'epochs': 10,
                         'src_vocab_length': len(SRC.vocab),
                         'target_vocab_length': len(TGT.vocab),
                         'num_layers': NUM_LAYERS,
                         'batch_size': BATCH_SIZE,
                         'model_size': model_size,
                         'factor': FACTOR,
                         'warmmup': WARMUP,
                         'learning_rate': LR,
                         'epsilon': EPSILON,
                         'max_steps': MAX_STEPS
                         })

    criterion = LabelSmoothing(
        size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()

    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(model_size, FACTOR, WARMUP,
                        torch.optim.Adam(model.parameters(), lr=LR, betas=(BETAS[0], BETAS[1]), eps=EPSILON))

    wandb.watch(model)

    current_steps = 0
    for epoch in range(10):
        model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), model_par,
                  MultiGPULossCompute(
                      model.generator, criterion, devices=devices, opt=model_opt),
                  epoch)

        model_par.eval()
        (loss, steps) = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                                  MultiGPULossCompute(
            model.generator, criterion, devices=devices, opt=None),
            epoch)
        print(loss)
        current_steps += steps
        if current_steps > MAX_STEPS:
            break

    current_date = datetime.now().strftime("%b-%d-%Y_%H-%M")
    torch.save(model.state_dict(), 'en-de__'+current_date+'.pt')

    validate(model, test_iter, SRC, TGT, BOS_WORD, EOS_WORD, BLANK_WORD)


main()
