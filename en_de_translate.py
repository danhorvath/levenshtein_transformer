import torch
import torch.nn as nn
from torchtext import data, datasets
from sacrebleu import corpus_bleu

from transformer.search import greedy_decode
from train import  run_epoch
from transformer.optimizer import NoamOpt
from transformer.criteria import LabelSmoothing
from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from transformer.model import Transformer
from transformer.data import batch_size_fn, MyIterator, rebatch
from validator import validate
from utils import save_model

from en_de_config import config

import wandb


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'


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
                                             # train='newstest2014.tok.bpe.32000',
                                             validation='newstest2013.tok.bpe.32000',
                                             test='newstest2014.tok.bpe.32000',
                                             fields=(SRC, TGT),
                                             filter_pred=lambda x: len(vars(x)['src']) <= config['max_len'] and
                                             len(vars(x)['trg']
                                                 ) <= config['max_len'],
                                             root='./.data/')
    print('Train set length: ', len(train))

    # building shared vocabulary
    TGT.build_vocab(train.src, train.trg, min_freq=config['min_freq'])
    SRC.vocab = TGT.vocab

    print('Source vocab length: ', len(SRC.vocab.itos))
    print('Target vocab length: ', len(TGT.vocab.itos))
    wandb.config.update({'src_vocab_length': len(SRC.vocab),
                         'target_vocab_length': len(TGT.vocab)})

    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    print('Pad index:', pad_idx)

    train_iter = MyIterator(train, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
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
        size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1, batch_multiplier=config['batch_multiplier'])
    criterion.cuda()

    eval_criterion = LabelSmoothing(
        size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1, batch_multiplier=1)
    eval_criterion.cuda()

    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(warmup_init_lr=config['warmup_init_lr'], warmup_end_lr=config['warmup_end_lr'], warmup_updates=config['warmup'],
                        optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(config['beta_1'], config['beta_2']), eps=config['epsilon']))

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
                                batch_multiplier=config['batch_multiplier'],
                                logging=True)

        current_steps += steps

        # calculating validation loss and bleu score
        model_par.eval()
        loss_calculator_without_optimizer = MultiGPULossCompute(
            model.generator, eval_criterion, devices=devices, opt=None)

        (loss, _) = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                              model_par,
                              loss_calculator_without_optimizer,
                              steps_so_far=current_steps)

        if (epoch > 10) or current_steps > config['max_step']:
            # greedy decoding takes a while so Bleu won't be evaluated for every epoch
            print('Calculating BLEU score...')
            bleu = validate(model, valid_iter, SRC, TGT,
                            BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'])
            wandb.log({'Epoch bleu': bleu})
            print(f'Epoch {epoch} | Bleu score: {bleu} ')

        print(f"Epoch {epoch} | Loss: {loss}")
        wandb.log({'Epoch': epoch, 'Epoch loss': loss})
        if epoch > 10:
            save_model(model=model, optimizer=model_opt, loss=loss, src_field=SRC, tgt_field=TGT, updates=current_steps, epoch=epoch)
        if current_steps > config['max_step']:
            break

    save_model(model=model, optimizer=model_opt, loss=loss, src_field=SRC, tgt_field=TGT, updates=current_steps, epoch=epoch)

    test_bleu = validate(model, test_iter, SRC, TGT,
                         BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'], logging=True)
    print(f"Test Bleu score: {test_bleu}")
    wandb.config.update({'Test bleu score': test_bleu})


main()
