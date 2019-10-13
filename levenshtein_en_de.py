import torch
import torch.nn as nn
from torchtext import data, datasets

from levenhtein_transformer.train import run_epoch
from transformer.optimizer import NoamOpt
from transformer.criterion import LabelSmoothingKLLoss
# from transformer.multi_gpu_loss_compute import MultiGPULossCompute
from levenhtein_transformer.model import LevenshteinTransformerModel
from transformer.data import MyIterator, batch_size_fn, rebatch_and_noise
# from transformer.model import Transformer
from transformer.data import batch_size_fn, MyIterator, rebatch
# from validator import validate
# from utils import save_model

from en_de_config import config

# import wandb


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = '<blank>'
UNK = '<unk>'


# wandb.init(project="levenshtein_transformer")
# wandb.config.update(config)


def main():
    devices = list(range(torch.cuda.device_count()))
    print('Selected devices: ', devices)

    def tokenize_bpe(text):
        return text.split()

    SRC = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD, unk_token=UNK)
    TGT = data.Field(tokenize=tokenize_bpe, init_token=BOS_WORD, unk_token=UNK,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    train, val, test = datasets.WMT14.splits(exts=('.en', '.de'),
                                             #  train='train.tok.clean.bpe.32000',
                                             train='newstest2014.tok.bpe.32000',
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
    # wandb.config.update({'src_vocab_length': len(SRC.vocab),
    #                      'target_vocab_length': len(TGT.vocab)})

    pad_idx = TGT.vocab.stoi[BLANK_WORD]
    bos_idx = TGT.vocab.stoi[BOS_WORD]
    eos_idx = TGT.vocab.stoi[EOS_WORD]
    unk_idx = TGT.vocab.stoi[UNK]
    print(f'Indexes -- PAD: {pad_idx}, EOS: {eos_idx}, BOS: {bos_idx}, UNK: {unk_idx}')

    train_iter = MyIterator(train, batch_size=4000, device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=True)

    valid_iter = MyIterator(val, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
                            sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    # test_iter = MyIterator(test, batch_size=config['batch_size'], device=torch.device(0), repeat=False,
    #                        sort_key=lambda x: (len(x.src), len(x.trg)), batch_size_fn=batch_size_fn, train=False)

    model = LevenshteinTransformerModel(len(SRC.vocab), len(TGT.vocab), N=4, PAD=pad_idx,
                                        BOS=bos_idx, EOS=eos_idx, UNK=unk_idx, d_model=512, d_ff=1024, h=2, dropout=0.1)

    # weight tying
    model.src_embed[0].lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.generator.lookup_table.weight = model.tgt_embed[0].lookup_table.weight
    model.cuda()

    model_size = model.src_embed[0].d_model
    print('Model created with size of', model_size)

    # wandb.config.update({'model_size': model_size})

    insertion_criterion = LabelSmoothingKLLoss(
        size=256, padding_idx=pad_idx, smoothing=0.1, batch_multiplier=config['batch_multiplier'])
    insertion_criterion.cuda()
    word_pred_criterion = LabelSmoothingKLLoss(
        size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1, batch_multiplier=config['batch_multiplier'])
    word_pred_criterion.cuda()
    del_criterion = LabelSmoothingKLLoss(
        size=2, padding_idx=pad_idx, smoothing=0.1, batch_multiplier=config['batch_multiplier'])
    del_criterion.cuda()
    # eval_criterion = LabelSmoothing(
    #     size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1, batch_multiplier=1)
    # eval_criterion.cuda()

    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = NoamOpt(warmup_init_lr=config['warmup_init_lr'], warmup_end_lr=config['warmup_end_lr'],
                        warmup_updates=config['warmup'],
                        optimizer=torch.optim.Adam(model.parameters(),
                                                   lr=0,
                                                   betas=(config['beta_1'], config['beta_2']),
                                                   eps=config['epsilon']))

    # wandb.watch(model)

    current_steps = 0
    for epoch in range(1, config['max_epochs'] + 1):
        # training model
        model_par.train()

        (_, steps) = run_epoch((rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in train_iter),
                               model=model_par,
                               criterions=[insertion_criterion, word_pred_criterion, del_criterion],
                               opt=model_opt,
                               steps_so_far=current_steps,
                               batch_multiplier=config['batch_multiplier'],
                               logging=True,
                               train=True)

        current_steps += steps

        # calculating validation loss and bleu score
        model_par.eval()

        (_, steps) = run_epoch((rebatch_and_noise(b, pad=pad_idx, bos=bos_idx, eos=eos_idx) for b in valid_iter),
                               model=model_par,
                               criterion=criterion,
                               opt=model_opt,
                               steps_so_far=current_steps,
                               batch_multiplier=config['batch_multiplier'],
                               logging=False,
                               train=False)

        # if (epoch > 10) or current_steps > config['max_step']:
        #     # greedy decoding takes a while so Bleu won't be evaluated for every epoch
        #     print('Calculating BLEU score...')
        #     bleu = validate(model, valid_iter, SRC, TGT,
        #                     BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'])
        #     wandb.log({'Epoch bleu': bleu})
        #     print(f'Epoch {epoch} | Bleu score: {bleu} ')
        #
        # print(f"Epoch {epoch} | Loss: {loss}")
        # wandb.log({'Epoch': epoch, 'Epoch loss': loss})
        # if epoch > 10:
        #     save_model(model=model, optimizer=model_opt, loss=loss, src_field=SRC, tgt_field=TGT, updates=current_steps, epoch=epoch)
        # if current_steps > config['max_step']:
        #     break

    # save_model(model=model, optimizer=model_opt, loss=loss, src_field=SRC, tgt_field=TGT, updates=current_steps, epoch=epoch)
    #
    # test_bleu = validate(model, test_iter, SRC, TGT,
    #                      BOS_WORD, EOS_WORD, BLANK_WORD, config['max_len'], logging=True)
    # print(f"Test Bleu score: {test_bleu}")
    # wandb.config.update({'Test bleu score': test_bleu})


main()
