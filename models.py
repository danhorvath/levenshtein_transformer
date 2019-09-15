import torch.nn as nn
from copy import deepcopy
from layers import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, EncoderDecoder, Encoder, \
    EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator


def Transformer(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    """
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout), N),
        Decoder(DecoderLayer(d_model, deepcopy(attn), deepcopy(attn), deepcopy(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), deepcopy(position)),
        Generator(d_model, tgt_vocab)
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
