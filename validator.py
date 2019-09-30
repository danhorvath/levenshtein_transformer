import wandb
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu as nltk_corpus_bleu
from sacrebleu import corpus_bleu
import numpy as np
from transformer.search import greedy_decode
from utils import vector_to_sentence, get_src_mask


def validate(model, valid_iter, SRC, TGT, BOS_WORD, EOS_WORD, BLANK_WORD, max_len, logging=False):
    print('Validating...')
    # TODO: paralell Bleu calculation

    batch_bleus = []
    batch_sacrebleus = []
    hypotheses_tokenized = []
    references_tokenized =[]

    hypotheses = []
    references =[]

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)
        tgt = batch.trg.transpose(0, 1)

        src_sentences = [vector_to_sentence(
            src[i, :], SRC, 0) for i in range(src.size(0))]

        tgt_sentences = [vector_to_sentence(tgt[i, :], TGT, EOS_WORD)
                         for i in range(tgt.size(0))]

        out_seqs = [greedy_decode(model, src_sentence.unsqueeze(-2), get_src_mask(src_sentence.unsqueeze(-2), SRC.vocab.stoi[BLANK_WORD]), max_len=max_len,
                                  start_symbol=TGT.vocab.stoi[BOS_WORD], stop_symbol=TGT.vocab.stoi[EOS_WORD]) for src_sentence in src]

        out_sentences = [vector_to_sentence(out_seq.squeeze(), TGT, EOS_WORD)
             for out_seq in out_seqs]

        hypotheses_tokenized += [out_sentence.split(' ') for out_sentence in out_sentences]
        references_tokenized += [[tgt_sentence.split(' ')] for tgt_sentence in tgt_sentences]

        hypotheses += out_sentences
        references += tgt_sentences

        # sentence_pairs = list(zip(tgt_sentences, out_sentences))
        if logging:
            # sentence_pair = list(sentence_pairs)[0]

            print(
                f"Source: {src_sentences[0]}\nTarget: {tgt_sentences[0]}\nPrediction: {out_sentences[0]}\n")

        # batch_bleu = [sentence_bleu([sentence_pair[0].split(' ')], sentence_pair[1].split(' '))
        #                     for sentence_pair in sentence_pairs]

        # batch_sacrebleu = [corpus_bleu(sentence_pair[1], sentence_pair[0]).score
        #                     for sentence_pair in sentence_pairs]

        # batch_bleus += batch_bleu
        # batch_sacrebleus += batch_sacrebleu

        # print(f'Batch {i} | Bleu: {np.array(batch_bleu).mean() * 100} | Sacrebleu: {np.array(batch_sacrebleu).mean()}')

    # bleu_score = np.array(batch_bleus).mean() * 100
    # sacrebleu_score = np.array(batch_sacrebleus).mean()
    # print(f'Avg bleu: {bleu_score} | Avg. sacrebleu: {sacrebleu_score}')
    corpus_sacrebleu_score = corpus_bleu(hypotheses, [references])
    corpus_bleu_score = nltk_corpus_bleu(references_tokenized, hypotheses_tokenized)
    print(f'Corpus bleu: {corpus_bleu_score * 100} | Corpus sacrebleu: {corpus_sacrebleu_score.score}')
    return corpus_sacrebleu_score.score
