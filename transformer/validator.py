import wandb
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from sacrebleu import corpus_bleu
from transformer.search import greedy_decode
from utils import vector_to_sentence, get_src_mask


def validate(model, valid_iter, SRC, TGT, BOS_WORD, EOS_WORD, BLANK_WORD, max_len=None, logging=False, is_test=False):
    print('Validating...')
    # TODO: paralell Bleu calculation

    hypotheses_tokenized = []
    references_tokenized = []

    hypotheses = []
    references = []

    table = wandb.Table(columns=["Source", "Target", "Prediction"])

    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)
        tgt = batch.trg.transpose(0, 1)
        if max_len is None:
            max_len = len(tgt[0]) * 2

        src_sentences = [vector_to_sentence(src[i, :], field=SRC, eos_word=EOS_WORD, start_from=0)
                         for i in range(src.size(0))]

        tgt_sentences = [vector_to_sentence(tgt[i, :], field=TGT, eos_word=EOS_WORD) for i in range(tgt.size(0))]

        out_seqs = [greedy_decode(model, src_sentence.unsqueeze(-2),
                                  get_src_mask(src_sentence.unsqueeze(-2), SRC.vocab.stoi[BLANK_WORD]), max_len=max_len,
                                  start_symbol=TGT.vocab.stoi[BOS_WORD], stop_symbol=TGT.vocab.stoi[EOS_WORD]) for
                    src_sentence in src]

        out_sentences = [vector_to_sentence(out_seq.squeeze(), TGT, EOS_WORD) for out_seq in out_seqs]

        hypotheses_tokenized += [out_sentence.split(' ') for out_sentence in out_sentences]
        references_tokenized += [[tgt_sentence.split(' ')] for tgt_sentence in tgt_sentences]

        hypotheses += out_sentences
        references += tgt_sentences

        table.add_data(src_sentences[0], tgt_sentences[0], out_sentences[0])
        if logging:
            print(f"Source: {src_sentences[0]}\nTarget: {tgt_sentences[0]}\nPrediction: {out_sentences[0]}\n")

    corpus_sacrebleu_score = corpus_bleu(hypotheses, [references])
    corpus_bleu_score = nltk_corpus_bleu(references_tokenized, hypotheses_tokenized)
    wandb.log({"Test samples" if is_test else "Validation samples": table})
    print(f'Corpus bleu: {corpus_bleu_score * 100} | Corpus sacrebleu: {corpus_sacrebleu_score.score}')
    return corpus_sacrebleu_score.score
