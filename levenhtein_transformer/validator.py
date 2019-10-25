import wandb
import torch
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from sacrebleu import corpus_bleu
from levenhtein_transformer.config import config
from levenhtein_transformer.utils import initialize_output_tokens
from utils import vector_to_sentence


def validate(model, iterator, bos, eos, max_decode_iter=10, logging=False, is_test=False):
    model.eval()
    for i, batch in enumerate(iterator):
        decode_iter = 0
        prev_out = initialize_output_tokens(batch.noised_trg, bos=bos, eos=eos)
        encoder_out = model.encode(batch.src, batch.src_mask)
        out = torch.tensor([[]])
        while decode_iter < max_decode_iter:
            out = model.decode(encoder_out, prev_out, batch.src_mask, max_ratio=config['max_decoder_ratio'])

            if out.tolist() == prev_out.tolist():
                break

            decode_iter += 1
            prev_out = out

        print('SRC', batch.src)
        print('OUT', out)

    return 1
