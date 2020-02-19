#!/usr/bin/env python
# encoding: utf-8

"""
@author: czhang
@contact: 
@Time : 2019/11/16 11:30
"""

import torch
import torch.nn.functional as F



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






























































