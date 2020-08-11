#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, f, e_char, k=5, padding=1):
        """
        @param f: the number of convolution kernels, equals to word embedding dimensions e_word
        @param e_char: character embedding dimensions
        @param k: kernel size
        @param padding: padding size
        """
        super().__init__()
        self.e_char = e_char
        self.e_word = f
        self.conv1d = nn.Conv1d(in_channels = e_char, out_channels = f, kernel_size = k, padding = 1)
        self.relu = nn.ReLU()
    
    def forward(self, char_embeds):
        """
        @param char_embeds: tensor with shape (max_sentence_length, batch_size, e_char, max_word_length)
        @returns x_conv_out: tensor with shape (max_sentence_length, batch_size, e_word)
        """
        max_sentence_length, batch_size, _, max_word_length = char_embeds.size()
        char_embeds = char_embeds.view(-1, self.e_char, max_word_length)
        x_conv_out = self.relu(self.conv1d(char_embeds))
        x_conv_out, _ = torch.max(x_conv_out, dim = -1)
        x_conv_out = x_conv_out.view(max_sentence_length, batch_size, self.e_word)

        return x_conv_out

    ### END YOUR CODE

if __name__ == '__main__':
    print("custom cnn sanity check!")

    char_embeds = torch.randn(12, 64, 256, 15)
    cnn = CNN(512, 256)
    x_conv_out = cnn(char_embeds)
    if x_conv_out.shape == (12, 64, 512):
        print("Output shape check pass!")


