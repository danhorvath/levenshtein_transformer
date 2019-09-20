import nltk
import numpy as np
from functions import greedy_decode
import wandb


def validate(model, valid_iter, SRC, TGT, BOS_WORD, EOS_WORD, BLANK_WORD):
    def bpe_to_words(sentence):
        new_sentence = []
        for i in range(len(sentence)):
            word = sentence[i]
            if word[-2:] == '@@' and i != len(sentence)-1:
                sentence[i+1] = word[:-2]+sentence[i+1]
            else:
                new_sentence.append(word)
        return new_sentence

    def vector_to_sentence(vector):
        sentence = []
        for l in range(1, len(vector)):
            word = TGT.vocab.itos[vector[l]]
            if word == EOS_WORD:
                break
            sentence.append(word)

        return bpe_to_words(sentence)

    def get_weights(sentence):
        length = len(sentence) if len(sentence) <= 4 else 4
        return list(np.ones(length)/length)

    def get_BLEU():
        bleu_scores = []
        for i, batch in enumerate(valid_iter):
            src = batch.src.transpose(0, 1)[:1]
            src_mask = (src != SRC.vocab.stoi[BLANK_WORD]).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, max_len=60,
                                start_symbol=TGT.vocab.stoi[BOS_WORD])

            predictions = [vector_to_sentence(out[i, :])
                           for i in range(out.size(0))]

            targets = [vector_to_sentence(batch.trg.data[:, i])
                       for i in range(batch.trg.size(1))]

            sentence_pairs = zip(targets, predictions)
            batch_bleu = np.array([nltk.translate.bleu_score.sentence_bleu(
                [sentence_pair[0]], sentence_pair[1], get_weights(sentence_pair[0])) for sentence_pair in sentence_pairs])
            bleu_scores.append(batch_bleu.mean())

        bleu_score = np.array(bleu_scores).mean()
        print('BLEU score: ', bleu_score)
        wandb.log({'BLEU': bleu_score})

    return get_BLEU()
