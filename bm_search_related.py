#!/usr/bin/env python
# encoding: utf-8

"""
@author: czhang
@contact: 
@Time : 2019/11/16 11:30
"""

import torch
import torch.nn.functional as F

use_cuda = True
embedding_size = 128
hidden_size = 512
vocab_size = 5000
SOS_ID = 1
EOS_ID = 2
max_length = 20

def beam_search(decoder, source_lengths, encoder_outputs, decoder_input, decoder_hidden, stoi, beam_size=3, max_length=10):
    """

    :param decoder:
    :param source_lengths:
    :param encoder_outputs:
    :param decoder_input:
    :param decoder_hidden:
    :param stoi: convert  sentence to id token
    :param beam_size:
    :param max_length:
    :return:
    """
    end_sentences = []
    for t in range(max_length):
        decoder_output, decoder_hidden = decoder.forward(decoder_input, decoder_hidden, encoder_outputs, source_lengths)
        output_logprob = F.log_softmax(decoder_output, dim=-1)
        topk_prob, topk_index = torch.topk(output_logprob, beam_size, -1)
        topk_prob = topk_prob.view(1, -1)
        topk_index = topk_index.view(1, -1)

        if t== 0:
            topk_prob_cat = topk_prob
            topk_index_cat = topk_index
            topk_score_cat = topk_prob_cat.mean(dim=0)
            decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
            encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)
        else:
            topk_prob_cat = topk_prob_cat.view(-1, 1).repeat(1, beam_size).view(t, -1)
            topk_index_cat = topk_index_cat.view(-1, 1).repeat(1, beam_size).view(t, -1)
            topk_prob_cat = torch.cat((topk_prob_cat, topk_prob), 0)
            topk_index_cat = torch.cat((topk_index_cat, topk_index), 0)
            topk_score_cat = topk_prob_cat.sum(dim=0)

        end_idx = (topk_index == 2).nonzero()

        if len(end_idx) > 0:
            end_idx = end_idx[:, 1]
            for l in range(len(end_idx)):
                i = end_idx[l]
                temp_score = topk_prob_cat.mean(dim=0)[i].item()
                temp_sentence = topk_index_cat[:, i]
                end_sentences.append((topk_index_cat[:, i].clone(), temp_score))
        next_idx = (topk_index != 2).nonzero()

        if len(next_idx) > 0:
            next_idx = next_idx[:, 1]
            if len(next_idx) > beam_size:
                temp = torch.topk(topk_score_cat[next_idx], beam_size)[1]
                next_idx = next_idx[temp]

                decoder_hidden = decoder_hidden[:, next_idx/beam_size, :]
                encoder_outputs = encoder_outputs[:, next_idx/beam_size, :]
                decoder_input = topk_index[:, next_idx]
                topk_index_cat = topk_index_cat[:, next_idx]
                topk_prob_cat = topk_prob_cat[:, next_idx]

            else:
                break
        end_sentences.sort(key=lambda s: s[1], reverse=True)
        end_sentence = end_sentences[0][0]
        end_lengths = torch.tensor(end_sentence.size(0))
        end_scores = end_sentences[0][1]

        return end_sentence, end_lengths, end_scores

































































