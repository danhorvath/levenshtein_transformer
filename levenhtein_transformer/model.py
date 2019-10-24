import torch.nn as nn
from copy import deepcopy
from transformer.layers import PositionalEncoding, EncoderLayer, Encoder, \
    DecoderLayer, Generator, Embeddings
from transformer.sublayers import MultiHeadedAttention, PositionwiseFeedForward
from levenhtein_transformer.layers import LevenshteinEncodeDecoder, LevenshteinDecoder


def LevenshteinTransformerModel(src_vocab, tgt_vocab, PAD, BOS, EOS, UNK, criterion, d_model=512, n=6, h=8, d_ff=2048,
                                dropout=0.0, input_dropout=0.1):
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, input_dropout)
    model = LevenshteinEncodeDecoder(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), n),
        LevenshteinDecoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout),
                           n=n, output_embed_dim=d_model, tgt_vocab=tgt_vocab),
        nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
        Generator(d_model, tgt_vocab),
        pad=PAD, bos=BOS, eos=EOS, unk=UNK,
        criterion=criterion
    )
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
