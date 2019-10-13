# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.nn as nn

from transformer.layers import Decoder, EncoderDecoder
from levenhtein_transformer.utils import _apply_del_words, _apply_ins_masks, _apply_ins_words, \
    _get_del_targets, _get_ins_targets, fill_tensors as _fill, skip_tensors as _skip
from transformer.data import BatchWithNoise


class LevenshteinEncodeDecoder(EncoderDecoder):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, pad, bos, eos, unk):
        super(LevenshteinEncodeDecoder, self).__init__(encoder, decoder, src_embed, tgt_embed, generator)
        self.pad = pad
        self.eos = eos
        self.bos = bos
        self.unk = unk

    def forward(self, src: torch.Tensor, x: torch.Tensor, src_mask: torch.Tensor, x_mask: torch.Tensor,
                tgt: torch.Tensor):

        assert tgt is not None, "Forward function only supports training."

        # encoding
        encoder_out = self.encode(src, src_mask)
        # print('src:', src[0], src.size())
        # print('tgt:', tgt[0], tgt.size())
        # print('x:', x[0], x.size())

        # generate training labels for insertion
        # word_pred_tgt, word_pred_tgt_masks, ins_targets
        word_pred_tgt, word_pred_tgt_masks, ins_targets = _get_ins_targets(x, tgt, self.pad, self.unk)
        word_pred_tgt_subsequent_masks = BatchWithNoise.make_std_mask(word_pred_tgt, pad=self.pad)
        # print('word_pred_tgt', word_pred_tgt[0], word_pred_tgt.size())
        # print('word_pred_tgt_subsequent_masks', word_pred_tgt_subsequent_masks[0],
        #       word_pred_tgt_subsequent_masks.size())
        # print('word_pred_tgt_masks', word_pred_tgt_masks[0], word_pred_tgt_masks.size())
        # print('ins_targets', ins_targets[0], ins_targets.size())

        ins_targets = ins_targets.clamp(min=0, max=255)  # for safe prediction
        # print('clamped ins_targets', ins_targets[0], ins_targets.size())
        ins_masks = x[:, 1:].ne(self.pad)
        # print('ins_masks', ins_masks[0], ins_masks.size())

        ins_out = self.decoder.forward_mask_ins(encoder_out, src_mask, self.tgt_embed(x), x_mask)
        # print('ins_out', ins_out[0], ins_out.size())

        word_pred_out = self.decoder.forward_word_ins(encoder_out, src_mask, self.tgt_embed(word_pred_tgt),
                                                      word_pred_tgt_subsequent_masks)
        # print('word_pred_out', word_pred_out[0], word_pred_out.size())

        # make online prediction
        word_predictions = F.log_softmax(word_pred_out, dim=-1).max(2)[1]
        word_predictions.masked_scatter_(~word_pred_tgt_masks, tgt[~word_pred_tgt_masks])
        word_predictions_subsequent_mask = BatchWithNoise.make_std_mask(word_predictions, pad=self.pad)
        # print('word_predictions', word_predictions[0], word_predictions.size())
        # print('word_predictions_subsequent_mask', word_predictions_subsequent_mask[0],
        #       word_predictions_subsequent_mask.size())

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt, self.pad)
        # print('word_del_targets', word_del_targets[0], word_del_targets.size())
        word_del_out = self.decoder.forward_word_del(encoder_out, src_mask, self.tgt_embed(word_predictions),
                                                     word_predictions_subsequent_mask)
        # print('word_del_out', word_del_out[0], word_del_out.size())

        return {
            "ins_out": ins_out,
            "ins_tgt": ins_targets,
            "ins_mask": ins_masks,
            "word_pred_out": word_pred_out,
            "word_pred_tgt": tgt,
            "word_pred_mask": word_pred_tgt_masks,
            "word_del_out": word_del_out,
            "word_del_tgt": word_del_targets,
            "word_del_mask": word_predictions.ne(self.pad),
        }

    def forward_encoder(self, src, src_mask):
        return self.self.encode(src, src_mask)

    def forward_decoder(self, output_tokens: torch.Tensor, encoder_out: torch.Tensor, eos_penalty=0.0,
                        max_ratio=None) -> torch.Tensor:

        # output_scores = decoder_out["output_scores"]
        output_scores = None

        if max_ratio is None:
            max_lens = output_tokens.new(output_tokens.size(0)).fill_(255)
        else:
            max_lens = ((~encoder_out["encoder_padding_mask"]).sum(1) * max_ratio).clamp(min=10)

        # delete words

        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2
        if can_del_word.sum() != 0:  # we cannot delete, skip

            word_del_out = self.decoder.forward_word_del(
                _skip(output_tokens, can_del_word), _skip(encoder_out, can_del_word)
            )
            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_pred = word_del_score.max(-1)[1].bool()

            _tokens = _apply_del_words(
                output_tokens[can_del_word],
                word_del_pred,
                self.pad,
                self.bos,
                self.eos,
            )

            output_tokens = _fill(output_tokens, can_del_word, _tokens, self.pad)

        # insert placeholders
        can_ins_mask = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_ins_mask.sum() != 0:
            mask_ins_out = self.decoder.forward_mask_ins(_skip(output_tokens, can_ins_mask),
                                                         _skip(encoder_out, can_ins_mask))
            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            if eos_penalty > 0.0:
                mask_ins_score[:, :, 0] -= eos_penalty
            mask_ins_pred = mask_ins_score.max(-1)[1]
            mask_ins_pred = torch.min(mask_ins_pred, max_lens[:, None].expand_as(mask_ins_pred))

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_ins_mask],
                mask_ins_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_ins_mask, _tokens, self.pad)

        # insert words
        can_ins_word = output_tokens.eq(self.unk).sum(1) > 0
        if can_ins_word.sum() != 0:
            word_ins_out = self.decoder.forward_word_ins(_skip(output_tokens, can_ins_word),
                                                         _skip(encoder_out, can_ins_word))
            word_ins_score = F.log_softmax(word_ins_out, 2)
            word_ins_pred = word_ins_score.max(-1)[1]

            _tokens = _apply_ins_words(
                output_tokens[can_ins_word],
                word_ins_pred,
                self.unk,
            )

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad)

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(1).max()
        output_tokens = output_tokens[:, :]
        output_scores = output_scores[:, :cut_off]
        return output_tokens

    def initialize_output_tokens(self, src_tokens: torch.Tensor):
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        return initial_output_tokens


class LevenshteinDecoder(Decoder):
    def __init__(self, layer, n, output_embed_dim, tgt_vocab):
        super(LevenshteinDecoder, self).__init__(layer, n)

        # embeds the number of tokens to be inserted, max 256
        self.embed_mask_ins = nn.Linear(output_embed_dim * 2, 256)

        # embeds the number of tokens to be inserted, max 256
        self.embed_word_pred = nn.Linear(output_embed_dim, tgt_vocab)

        # embeds either 0 or 1
        self.embed_word_del = nn.Linear(output_embed_dim, 2)

    def forward_mask_ins(self, encoder_out: torch.Tensor, encoder_out_mask: torch.Tensor, x: torch.Tensor,
                         x_mask: torch.Tensor):
        features = self.forward(x, encoder_out, encoder_out_mask, x_mask)
        # creates pairs of consecutive words, so if the words are marked by their indices (0, 1, ... n):
        # [
        #   [0, 1],
        #   [1, 2],
        #   ...
        #   [n-1, n],
        # ]

        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        return self.embed_mask_ins(features_cat)

    def forward_word_ins(self, encoder_out: torch.Tensor, encoder_out_mask: torch.Tensor, x: torch.Tensor,
                         x_mask: torch.Tensor):
        features = self.forward(x, encoder_out, encoder_out_mask, x_mask)
        return self.embed_word_pred(features)

    def forward_word_del(self, encoder_out: torch.Tensor, encoder_out_mask: torch.Tensor, x: torch.Tensor,
                         x_mask: torch.Tensor):
        features = self.forward(x, encoder_out, encoder_out_mask, x_mask)
        return self.embed_word_del(features)

    def forward_word_del_mask_ins(self, encoder_out: torch.Tensor, encoder_out_mask: torch.Tensor, x: torch.Tensor,
                                  x_mask: torch.Tensor):
        # merge the word-deletion and mask insertion into one operation,
        features = self.forward(x, encoder_out, encoder_out_mask, x_mask)
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        f_word_del = F.log_softmax(self.embed_word_del(features), dim=2)
        f_mask_ins = F.log_softmax(self.embed_mask_ins(features_cat), dim=2)
        return f_word_del, f_mask_ins
