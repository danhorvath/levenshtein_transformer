import torch
import torch.nn as nn
from torchtext import data, datasets

from transformer.optimizer import NoamOpt
from levenhtein_transformer.train import run_epoch
from levenhtein_transformer.criterion import LabelSmoothingLoss
from levenhtein_transformer.model import LevenshteinTransformerModel
from levenhtein_transformer.data import rebatch_and_noise, batch_size_fn, MyIterator
from levenhtein_transformer.validator import validate
from utils import save_model

from levenhtein_transformer.config import config

import wandb

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
UNK = '<unk>'

wandb.init(project="levenshtein_transformer")
wandb.config.update(config)


def main():
    devices = list(range(torch.cuda.device_count()))
    print('Selected devices: ', devices)

    def tokenize_bpe(text):
        return text.split()

    SRC = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD, unk_token=UNK)
    TGT = data.Field(tokenize=tokenize_bpe, init_token=BOS_WORD, unk_token=UNK,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                             train='train.tok.clean.bpe.32000',
                                            #  train='newstest2014.tok.bpe.32000',
                                             validation='newstest2013.tok.bpe.32000',
                                             test='newstest2014.tok.bpe.32000',
                                             fields=(SRC, TGT),
                                             filter_pred=lambda x: len(vars(x)['src']) <= config['max_len'] and
                                                                   len(vars(x)['trg']) <= config['max_len'],
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
    bos_idx = TGT.vocab.stoi[BOS_WORD]
    eos_idx = TGT.vocab.stoi[EOS_WORD]
    unk_idx = TGT.vocab.stoi[UNK]
    print(f'Indexes -- PAD: {pad_idx}, EOS: {eos_idx}, BOS: {bos_idx}, UNK: {unk_idx}')

    train_iter = MyIterator(train, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

    valid_iter = MyIterator(val, batch_size=config['val_batch_size'], device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    test_iter = MyIterator(test, batch_size=config['val_batch_size'], device=torch.device(0), repeat=False,
                           sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    criterion = LabelSmoothingLoss(label_smoothing=0.1, batch_multiplier=config['batch_multiplier'])
    criterion.cuda()

    # model = LevenshteinTransformerModel(len(SRC.vocab), len(TGT.vocab), n=1, PAD=pad_idx,
    #                                     BOS=bos_idx, EOS=eos_idx, UNK=unk_idx,
    #                                     criterion=criterion,
    #                                     d_model=256, d_ff=256, h=1,
    #                                     dropout=config['dropout'], decoder_dropout=config['decoder_dropout'])

    model = LevenshteinTransformerModel(len(SRC.vocab), len(TGT.vocab),
                                        n=config['num_layers'],
                                        h=config['attn_heads'],
                                        d_model=config['model_dim'],
                                        dropout=config['dropout'],
                                        decoder_dropout=config['decoder_dropout'],
                                        d_ff=config['ff_dim'],
                                        criterion=criterion,
                                        PAD=pad_idx, BOS=bos_idx, EOS=eos_idx, UNK=unk_idx)

    # weight tying
    model.src_embed[0].lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.generator.lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.cuda()

    model_size = model.src_embed[0].d_model
    print('Model created with size of', model_size)

    wandb.config.update({'model_size': model_size})

    class MyDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super(MyDataParallel, self).__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

    model_par = MyDataParallel(model, device_ids=devices)

    model_opt = NoamOpt(warmup_init_lr=config['warmup_init_lr'], warmup_end_lr=config['warmup_end_lr'],
                        warmup_updates=config['warmup'],
                        min_lr=config['min_lr'],
                        optimizer=torch.optim.Adam(model.parameters(),
                                                   lr=0.0005,
                                                   weight_decay=config['weight_decay'],
                                                   betas=(config['beta_1'], config['beta_2']),
                                                   eps=config['epsilon']))

    wandb.watch(model)

    current_steps = 0
    for epoch in range(1, config['max_epochs'] + 1):
        # training model
        model_par.train()

        (_, steps) = run_epoch((rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in train_iter),
                               model=model_par,
                               opt=model_opt,
                               steps_so_far=current_steps,
                               batch_multiplier=config['batch_multiplier'],
                               logging=True,
                               train=True)

        current_steps += steps

        # calculating validation loss and bleu score
        model_par.eval()

        (loss, _) = run_epoch((rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in valid_iter),
                              model=model_par,
                              opt=model_opt,
                              steps_so_far=current_steps,
                              batch_multiplier=config['batch_multiplier'],
                              logging=False,
                              train=False)

        wandb.log({'Epoch': epoch, 'Epoch loss': loss})
        if (epoch > 10) or current_steps > config['max_step']:
            print('Calculating BLEU score...')
            bleu = validate(model=model_par,
                            iterator=(rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in valid_iter),
                            SRC=SRC, TGT=TGT, EOS_WORD=EOS_WORD, bos=bos_idx, eos=eos_idx,
                            max_decode_iter=config['max_decode_iter'], logging=False)
            # wandb.log({'Epoch bleu': bleu})
            print(f'Epoch {epoch} | Bleu score: {bleu} ')

        print(f"Epoch {epoch} | Loss: {loss}")
        if epoch > 10:
            save_model(model=model, optimizer=model_opt, loss=loss, src_field=SRC, tgt_field=TGT, updates=current_steps,
                       epoch=epoch, prefix='lev_t')
        if current_steps > config['max_step']:
            break

    save_model(model=model, optimizer=model_opt, loss=loss, src_field=SRC, tgt_field=TGT, updates=current_steps,
               epoch=epoch, prefix='lev_t_final')

    test_bleu = validate(model=model_par,
                         iterator=(rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in test_iter),
                         SRC=SRC, TGT=TGT, EOS_WORD=EOS_WORD, bos=bos_idx, eos=eos_idx,
                         max_decode_iter=config['max_decode_iter'], logging=True)
    print(f"Test Bleu score: {test_bleu}")
    wandb.config.update({'Test bleu score': test_bleu})


main()
