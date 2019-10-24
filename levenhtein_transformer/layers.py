# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from transformer.layers import Decoder, EncoderDecoder
from levenhtein_transformer.utils import _apply_del_words, _apply_ins_masks, _apply_ins_words, \
    _get_del_targets, _get_ins_targets, fill_tensors as _fill, skip_tensors as _skip
from levenhtein_transformer.data import BatchWithNoise


class LevenshteinEncodeDecoder(EncoderDecoder):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad, bos, eos, unk, criterion):
        super(LevenshteinEncodeDecoder, self).__init__(encoder, decoder, src_embed, tgt_embed, generator)
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.unk = unk
        self.criterion = criterion

    def forward(self, src: Tensor, x: Tensor, src_mask: Tensor, x_mask: Tensor, tgt: Tensor):

        # TODO: CHECK OUTPUTS OF ALL TGT-PRED PAIRS!!!

        assert tgt is not None, "Forward function only supports training."

        # encoding
        encoder_out = self.encode(src, src_mask)

        # generate training labels for insertion
        # word_pred_tgt, word_pred_tgt_masks, ins_targets
        word_pred_input, word_pred_tgt_masks, ins_targets = _get_ins_targets(x, tgt, self.pad, self.unk)
        word_pred_tgt_subsequent_masks = BatchWithNoise.make_std_mask(word_pred_input, pad=self.pad)

        ins_targets = ins_targets.clamp(min=0, max=255)  # for safe prediction
        ins_masks = x[:, 1:].ne(self.pad)
        ins_out = self.decoder.forward_mask_ins(encoder_out, src_mask, self.tgt_embed(x), x_mask)

        word_pred_out = self.decoder.forward_word_ins(encoder_out, src_mask, self.tgt_embed(word_pred_input),
                                                      word_pred_tgt_subsequent_masks)
        # make online prediction
        word_predictions = F.log_softmax(word_pred_out, dim=-1).max(2)[1]
        word_predictions.masked_scatter_(~word_pred_tgt_masks, tgt[~word_pred_tgt_masks])
        word_predictions_subsequent_mask = BatchWithNoise.make_std_mask(word_predictions, pad=self.pad)

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt, self.pad)
        word_del_out = self.decoder.forward_word_del(encoder_out, src_mask, self.tgt_embed(word_predictions),
                                                     word_predictions_subsequent_mask)
        word_del_mask = word_predictions.ne(self.pad)

        ins_loss = self.criterion(outputs=ins_out, targets=ins_targets, masks=ins_masks, label_smoothing=0.0)
        word_pred_loss = self.criterion(outputs=word_pred_out, targets=tgt, masks=word_pred_tgt_masks,
                                        label_smoothing=0.1)
        word_del_loss = self.criterion(outputs=word_del_out, targets=word_del_targets,
                                       masks=word_del_mask, label_smoothing=0.01)

        return {
            "ins_out": ins_out,
            "ins_tgt": ins_targets,
            "ins_mask": ins_masks,
            "ins_loss": ins_loss,
            "word_pred_out": word_pred_out,
            "word_pred_tgt": tgt,
            "word_pred_mask": word_pred_tgt_masks,
            "word_pred_loss": word_pred_loss,
            "word_del_out": word_del_out,
            "word_del_tgt": word_del_targets,
            "word_del_mask": word_del_mask,
            "word_del_loss": word_del_loss,
            "loss": ins_loss + word_pred_loss + word_del_loss
        }

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, encoder_out: Tensor, x: Tensor, encoder_padding_mask: Tensor,
               eos_penalty=0.0, max_ratio=None) -> Tensor:

        if max_ratio is None:
            max_lens = x.new().fill_(255)
        else:
            src_lens = encoder_padding_mask.squeeze(1).sum(1)
            max_lens = (src_lens * max_ratio).clamp(min=10).long()

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = x.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip
            x_mask = BatchWithNoise.make_std_mask(x, self.pad)
            word_del_out = self.decoder.forward_word_del(
                _skip(encoder_out, can_del_word),
                _skip(encoder_padding_mask, can_del_word),
                self.tgt_embed(_skip(x, can_del_word)),
                _skip(x_mask, can_del_word))

            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens = _apply_del_words(
                x[can_del_word],
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )

            x = _fill(x, can_del_word, _tokens, self.pad)

        # insert placeholders
        can_ins_mask = x.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            x_mask = BatchWithNoise.make_std_mask(x, self.pad)
            mask_ins_out = self.decoder.forward_mask_ins(
                _skip(encoder_out, can_ins_mask),
                _skip(encoder_padding_mask, can_ins_mask),
                self.tgt_embed(_skip(x, can_ins_mask)),
                _skip(x_mask, can_ins_mask))

            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] -= eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(mask_ins_pred, max_lens[can_ins_mask, None].expand_as(mask_ins_pred))

            _tokens = _apply_ins_masks(
                x[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            x = _fill(x, can_ins_mask, _tokens, self.pad)

        # insert words
        can_ins_word = x.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            x_mask = BatchWithNoise.make_std_mask(x, self.pad)
            word_ins_out = self.decoder.forward_word_ins(
                _skip(encoder_out, can_ins_word),
                _skip(encoder_padding_mask, can_ins_word),
                self.tgt_embed(_skip(x, can_ins_word)),
                _skip(x_mask, can_ins_word))

            word_ins_score = F.log_softmax(word_ins_out, 2)
            word_ins_pred = word_ins_score.max(-1)[1]

            _tokens = _apply_ins_words(
                x[can_ins_word],
                word_ins_pred,
                self.unk,
            )

            x = _fill(x, can_ins_word, _tokens, self.pad)

        # delete some unnecessary paddings
        cut_off = x.ne(self.pad).sum(1).max()
        x = x[:, :cut_off]
        return x


class LevenshteinDecoder(Decoder):
    def __init__(self, layer, n, output_embed_dim, tgt_vocab):
        super(LevenshteinDecoder, self).__init__(layer, n)

        # embeds the number of tokens to be inserted, max 256
        self.embed_mask_ins = Embedding(256, output_embed_dim * 2, None)

        # embeds the number of tokens to be inserted, max 256
        self.embed_word_pred = nn.Parameter(torch.Tensor(tgt_vocab, output_embed_dim))
        self.sqrt_d_model = output_embed_dim ** -0.5

        # embeds either 0 or 1
        self.embed_word_del = Embedding(2, output_embed_dim, None)

    def extract_features(self, x, encoder_out, encoder_out_mask, x_mask):
        return self.forward(x, encoder_out, encoder_out_mask, x_mask)

    def forward_mask_ins(self, encoder_out: Tensor, encoder_out_mask: Tensor, x: Tensor, x_mask: Tensor):
        features = self.extract_features(x, encoder_out, encoder_out_mask, x_mask)
        # creates pairs of consecutive words, so if the words are marked by their indices (0, 1, ... n):
        # [
        #   [0, 1],
        #   [1, 2],
        #   ...
        #   [n-1, n],
        # ]

        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        return F.linear(features_cat, self.embed_mask_ins.weight) * self.sqrt_d_model

    def forward_word_ins(self, encoder_out: Tensor, encoder_out_mask: Tensor, x: Tensor, x_mask: Tensor):
        features = self.extract_features(x, encoder_out, encoder_out_mask, x_mask)
        return F.linear(features, self.embed_word_pred) * self.sqrt_d_model

    def forward_word_del(self, encoder_out: Tensor, encoder_out_mask: Tensor, x: Tensor, x_mask: Tensor):
        features = self.extract_features(x, encoder_out, encoder_out_mask, x_mask)
        return F.linear(features, self.embed_word_del.weight) * self.sqrt_d_model


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
