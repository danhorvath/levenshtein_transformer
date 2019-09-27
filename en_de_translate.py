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

import wandb


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'

config = {'max_epochs': 20,
          'num_layers': 6,
          'batch_size': 26000,
          'max_len': 150,
          'min_freq': 1,
          'factor': 1,
          'warmup': 4000,
          'lr': 0,
          'epsilon': 1e-9,
          'max_steps': 1e5,
          'beta_1': 0.9,
          'beta_2': 0.98
          }

wandb.init(project="transformer")
wandb.config.update(config)


def main():
    devices = list(range(torch.cuda.device_count()))
    print('Selected devices: ', devices)

    def tokenize_bpe(text):
        return text.split()

    SRC = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_bpe, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                                train='train.tok.clean.bpe.32000',
                                                validation='newstest2013.tok.bpe.32000',
                                                test='newstest2014.tok.bpe.32000',
                                                fields=(SRC, TGT),
                                                filter_pred=lambda x: len(vars(x)['src']) <= config['max_len'] and
                                                                    len(vars(x)['trg']) <= config['max_len'],
                                                root = './.data/')
    print('Train set length: ', len(train))

    # building shared vocabulary
    TGT.build_vocab(train.src, train.trg, min_freq = config['min_freq'])
    SRC.vocab=TGT.vocab

    print('Source vocab length: ', len(SRC.vocab.itos))
    print('Target vocab length: ', len(TGT.vocab.itos))
    wandb.config.update({'src_vocab_length': len(SRC.vocab),
                         'target_vocab_length': len(TGT.vocab)})

    pad_idx=TGT.vocab.stoi[BLANK_WORD]
    print('Pad index:', pad_idx)

    train_iter=MyIterator(train, batch_size = config['batch_size'], device = torch.device(0), repeat = False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

    valid_iter = MyIterator(val, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    test_iter = MyIterator(test, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
                           sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    model = Transformer(len(SRC.vocab), len(TGT.vocab), N=config['num_layers'])

    # weight tying
    model.src_embed[0].lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.generator.lookup_table.weight = model.tgt_embed[0].lookup_table.weight

    model.cuda()

    model_size = model.src_embed[0].d_model
    print('Model created with size of', model_size)
    wandb.config.update({'model_size': model_size})

    criterion = LabelSmoothing(
        size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()

    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(model_size, config['factor'], config['warmup'],
                        torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta_1'], config['beta_2']), eps=config['epsilon']))

    wandb.watch(model)

    current_steps = 0
    for epoch in range(1, config['max_epochs']+1):
        # training model
        model_par.train()
        loss_calculator = MultiGPULossCompute(
            model.generator, criterion, devices=devices, opt=model_opt)

        (_,  steps) = run_epoch((rebatch(pad_idx, b) for b in train_iter),
                                model_par,
                                loss_calculator,
                                steps_so_far=current_steps,
                                logging=True)

        current_steps += steps

        # calculating validation loss and bleu score
        model_par.eval()
        loss_calculator_without_optimizer = MultiGPULossCompute(
            model.generator, criterion, devices=devices, opt=None)

        (loss, _) = run_epoch((rebatch(pad_idx, b) for b in valid_iter), model_par,
                              loss_calculator_without_optimizer,
                              current_steps)

        if (epoch > 5 and epoch%2==1) or current_steps > config['max_steps']:
            # greedy decoding takes a while so Bleu won't be evaluated for every epoch
            bleu = validate(model, valid_iter, SRC, TGT,
                            BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'])
            wandb.log({'Epoch bleu': bleu})
            print(f'Epoch {epoch} | Bleu score: {bleu} ')
            
        print(f"Epoch {epoch} | Loss: {loss}")
        wandb.log({'Epoch': epoch, 'Epoch loss': loss})
        if epoch % 5 == 0 and epoch != 0:
            current_date = datetime.now().strftime("%b-%d-%Y_%H-%M")
            torch.save(model.state_dict(), 'en-de__'+current_date+'.pt')
        if current_steps > config['max_steps']:
            break

    current_date = datetime.now().strftime("%b-%d-%Y_%H-%M")
    file_name = 'en-de__'+current_date+'.pt'
    torch.save(model.state_dict(), file_name)
    print(f'Model saved as {file_name}')

    test_bleu = validate(model, test_iter, SRC, TGT,
                         BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'], logging=True)
    print(f"Test Bleu score: {test_bleu}")
    wandb.config.update({'Test bleu score': test_bleu})


main()
