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



devices = list(range(torch.cuda.device_count()))
print('Selected devices: ', devices)

def tokenize_bpe(text):
    return text.split()

TGT=torch.load('SRC_36893', pickle_module=dill)
SRC=torch.load('TGT_36893', pickle_module=dill)

train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                            # train='train.tok.clean.bpe.32000',
                                            train='newstest2014.tok.bpe.32000',
                                            validation='newstest2014.tok.bpe.32000',
                                            test='newstest2014.tok.bpe.32000',
                                            fields=(SRC, TGT),
                                            filter_pred=lambda x: len(vars(x)['src']) <= config['max_len'] and
                                            len(vars(x)['trg']
                                                ) <= config['max_len'],
                                            root='./.data/')
print('Train set length: ', len(train))



print('Source vocab length: ', len(SRC.vocab.itos))
print('Target vocab length: ', len(TGT.vocab.itos))

test_iter = MyIterator(test, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)


model = Transformer(len(SRC.vocab), len(TGT.vocab), N=config['num_layers'])


model.load_state_dict(torch.load('en-de__Sep-30-2019_09-19.pt'))
print('Last model loaded.')
save_model(model, None, loss=0, src_field=SRC, tgt_field=TGT, updates=0, epoch=0, prefix='last_model')

model.cuda()
model.eval()

# test_bleu = validate(model, test_iter, SRC, TGT,
#                         BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'], logging=True)
# print(f"Test Bleu score: {test_bleu}")



paths = ['en-de__Sep-30-2019_09-19.pt', 'en-de__Sep-29-2019_15-35.pt', 'en-de__Sep-29-2019_19-13.pt', 'en-de__Sep-29-2019_23-55.pt', 'en-de__Sep-30-2019_04-37.pt']
average_model_state = average_checkpoints(paths)

model.load_state_dict(average_model_state)
print(f'Model loaded with with averaged parameters from {len(paths)} models.')
save_model(model, None, loss=0, src_field=SRC, tgt_field=TGT, updates=0, epoch=0, prefix='average')

model.cuda()
model.eval()

# test_bleu = validate(model, test_iter, SRC, TGT,
#                         BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'], logging=True)
# print(f"Test Bleu score: {test_bleu}")

