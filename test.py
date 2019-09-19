from functions import greedy_decode
from data import MyIterator, batch_size_fn
from models import Transformer
import spacy
from torchtext import data, datasets
import torch


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

model = Transformer(len(SRC.vocab), len(TGT.vocab), N=NUM_LAYERS)

print('Model ready')


def test_model(model, test_iter, SRC, TGT):
    for i, batch in enumerate(test_iter):
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


TEST_ITER = MyIterator(test, batch_size=BATCH_SIZE, device=torch.device(0), repeat=False,
                       sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

test_model(model, test, SRC, TGT)
