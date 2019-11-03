import torch
import torch.nn as nn
from torchtext import data, datasets
import spacy

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

wandb.init(project="levenshtein_transformer_multi30k")
wandb.config.update(config)


def main():
    devices = list(range(torch.cuda.device_count()))
    print('Selected devices: ', devices)

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = data.Field(tokenize=tokenize_en, pad_token=BLANK_WORD, unk_token=UNK)
    TGT = data.Field(tokenize=tokenize_de, init_token=BOS_WORD, unk_token=UNK,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    train, val, test = datasets.Multi30k.splits(exts=('.en', '.de'),
                                                train='train',
                                                validation='val',
                                                test='test2016',
                                                fields=(SRC, TGT),
                                                filter_pred=lambda x: len(vars(x)['src']) <= config['max_len'] and
                                                                      len(vars(x)['trg']) <= config['max_len'],
                                                root='./.data/')

    print('Train set length: ', len(train))
    wandb.config.update({'Train set length': len(train)})

    # building shared vocabulary
    TGT.build_vocab(train.src, train.trg, min_freq=config['min_freq'])
    SRC.vocab = TGT.vocab

    print('Source vocab length: ', len(SRC.vocab.itos))
    print('Target vocab length: ', len(TGT.vocab.itos))
    wandb.config.update({'Source vocab length': len(SRC.vocab), 'Target vocab length': len(TGT.vocab)})

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

    criterion = LabelSmoothingLoss(batch_multiplier=config['batch_multiplier'])
    criterion.cuda()

    # model = LevenshteinTransformerModel(len(SRC.vocab), len(TGT.vocab), n=1, PAD=pad_idx,
    #                                     BOS=bos_idx, EOS=eos_idx, UNK=unk_idx,
    #                                     criterion=criterion,
    #                                     d_model=256, d_ff=256, h=1,
    #                                     dropout=config['dropout'],
    #                                     input_dropout=config['input_dropout'])

    model = LevenshteinTransformerModel(len(SRC.vocab), len(TGT.vocab),
                                        n=config['num_layers'],
                                        h=config['attn_heads'],
                                        d_model=config['model_dim'],
                                        dropout=config['dropout'],
                                        input_dropout=config['input_dropout'],
                                        d_ff=config['ff_dim'],
                                        criterion=criterion,
                                        PAD=pad_idx, BOS=bos_idx, EOS=eos_idx, UNK=unk_idx)

    # weight tying
    model.src_embed[0].lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.generator.lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.decoder.out_layer.lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.cuda()

    model_size = model.src_embed[0].d_model
    print('Model created with size of', model_size)

    wandb.config.update({'Model size': model_size})

    # make the inner model functions available from the DataParallel wrapper
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
                                                   lr=0,
                                                   weight_decay=config['weight_decay'],
                                                   betas=(config['beta_1'], config['beta_2']),
                                                   eps=config['epsilon']))

    wandb.watch(model)

    current_steps = 0
    epoch = 0
    while True:
        # training model
        print('Epoch ', epoch)
        wandb.log({'Epoch': epoch}, commit=False)
        model_par.train()

        loss, steps = run_epoch((rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in train_iter),
                                model=model_par,
                                opt=model_opt,
                                steps_so_far=current_steps,
                                batch_multiplier=config['batch_multiplier'],
                                logging=True,
                                train=True)

        current_steps += steps

        # calculating validation bleu score
        model_par.eval()
        bleu = validate(model=model_par,
                        iterator=(rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in valid_iter),
                        SRC=SRC, TGT=TGT, EOS_WORD=EOS_WORD, bos=bos_idx, eos=eos_idx, pad=pad_idx,
                        max_decode_iter=min(epoch + 1, config['max_decode_iter']), logging=False)
        wandb.log({'Epoch bleu score': bleu}, commit=False)
        if current_steps > config['max_step']:
            break
        epoch += 1

    save_model(model=model, optimizer=model_opt.optimizer, loss=loss, src_field=SRC, tgt_field=TGT,
               updates=current_steps, epoch=epoch, prefix=f'lev_t_epoch_final___')

    test_bleu = validate(model=model_par,
                         iterator=(rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in test_iter),
                         SRC=SRC, TGT=TGT, EOS_WORD=EOS_WORD, bos=bos_idx, eos=eos_idx, pad=pad_idx,
                         max_decode_iter=config['max_decode_iter'], logging=False, is_test=True)
    print(f"Test Bleu score: {test_bleu}")
    wandb.config.update({'Test bleu score': test_bleu})


main()
