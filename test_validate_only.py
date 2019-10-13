import torch
import torch.nn as nn
from torchtext import data, datasets


from transformer.search import greedy_decode
from train import  run_epoch
from transformer.optimizer import NoamOpt
from transformer.criterion import LabelSmoothingKLLoss
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.model import Transformer
from transformer.data import batch_size_fn, MyIterator, rebatch
from validator import validate
from utils import save_model, load_models, average_checkpoints

from en_de_config import config
import dill


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'

PATH = 'last_model_en-de__Sep-30-2019_18-19___old.pt'
print('Testing model ' + PATH)

state = torch.load(PATH)


def tokenize_bpe(text):
    return text.split()

SRC = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD)
TGT = data.Field(tokenize=tokenize_bpe, init_token=BOS_WORD,
                    eos_token=EOS_WORD, pad_token=BLANK_WORD)

TGT=torch.load('SRC_36893.pt', pickle_module=dill)
SRC=torch.load('TGT_36893.pt', pickle_module=dill)

train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                            train='newstest2014.tok.bpe.32000',
                                            validation='newstest2014.tok.bpe.32000',
                                            test='newstest2014.tok.bpe.32000',
                                            fields=(SRC, TGT),
                                            filter_pred=lambda x: len(vars(x)['src']) <= config['max_len'] and
                                            len(vars(x)['trg']
                                                ) <= config['max_len'],
                                            root='./.data/')

# building shared vocabulary

test_iter = MyIterator(test, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)


model = Transformer(len(SRC.vocab), len(TGT.vocab), N=config['num_layers'])
model.load_state_dict(state['model_state_dict'])
print('Model loaded.')

model.cuda()
model.eval()

test_bleu = validate(model, test_iter, SRC, TGT,
                        BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'], logging=True)
print(f"Test Bleu score: {test_bleu}")