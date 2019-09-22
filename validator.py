import nltk
import numpy as np
from functions import greedy_decode
import wandb


def validate(model, valid_iter, SRC, TGT, BOS_WORD, EOS_WORD, BLANK_WORD, logging=False):
    def bpe_to_words(sentence):
        new_sentence = []
        for i in range(len(sentence)):
            word = sentence[i]
            if word[-2:] == '@@' and i != len(sentence)-1:
                sentence[i+1] = word[:-2]+sentence[i+1]
            else:
                new_sentence.append(word)
        return new_sentence

    def vector_to_sentence(vector, field, start_from=1):
        sentence = []
        for l in range(start_from, len(vector)):
            word = field.vocab.itos[vector[l]]
            if word == EOS_WORD: break
            sentence.append(word)
            
        return bpe_to_words(sentence)

    def get_weights(sentence):
        length = len(sentence) if len(sentence) <= 4 else 4
        return list(np.ones(length)/length)

    def get_src_mask(src):
        return (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)

    def get_BLEU():
        bleu_scores = []
        for i, batch in enumerate(valid_iter):
            src = batch.src.transpose(0, 1)
            tgt = batch.trg.transpose(0, 1)

            src_sentences = [vector_to_sentence(src[i, :], SRC, 0) for i in range(src.size(0))]

            tgt_sentences = [vector_to_sentence(tgt[i, :], TGT) for i in range(tgt.size(0))]

            out_seqs = [greedy_decode(model, src_sentence.unsqueeze(-2), get_src_mask(src_sentence.unsqueeze(-2)), max_len=60, start_symbol=TGT.vocab.stoi[BOS_WORD]) for src_sentence in src]

            out_sentences = [vector_to_sentence(out_seq.squeeze(), TGT) for out_seq in out_seqs]

            sentence_pairs = zip(tgt_sentences, out_sentences)
            if logging:
                sentence_pair = list(sentence_pairs)[0]

                print(f"Source: {(' '.join(src_sentences[0])).encode('utf-8').decode('latin-1')}\n Target: {(' '.join(sentence_pair[0])).encode('utf-8').decode('latin-1')}\n Prediction: {(' '.join(sentence_pair[1])).encode('utf-8').decode('latin-1')}\n")

            batch_bleu = np.array([nltk.translate.bleu_score.sentence_bleu(
                [sentence_pair[0]], sentence_pair[1], get_weights(sentence_pair[0])) for sentence_pair in sentence_pairs])
            bleu_scores.append(batch_bleu.mean())

        bleu_score = np.array(bleu_scores).mean()
        wandb.log({'BLEU': bleu_score})
        return bleu_score

    return get_BLEU()
