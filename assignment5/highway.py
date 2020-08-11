#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, e_word):
        """
        @param e_word: the word embedding dimensions
        """
        super().__init__()
        self.e_word = e_word
        self.Linear1 = nn.Linear(in_features = e_word, out_features = e_word, bias = True)
        self.Linear2 = nn.Linear(in_features = e_word, out_features = e_word, bias = True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_conv_out):
        """
        @param x_conv_out: tensor with shape (max_sentence_length, batch_size, e_word)
        @returns x_highway: tensor with same shape as x_conv_out
        """

        x_proj = self.relu(self.Linear1(x_conv_out))
        x_gate = self.sigmoid(self.Linear2(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out

        return x_highway


    ### END YOUR CODE

if __name__ == '__main__':
    print("Custom highway network sanity Check!!")

    #output shape check
    e_word = 128
    x_conv_out = torch.randn(64, 20, 128)
    higway = Highway(e_word)
    x_highway = higway(x_conv_out)
    if x_conv_out.shape == x_highway.shape:
        print("output Shape check pass")
    
    #edge cas test, x_conv_out = 0, the expected x_highway should be 0 too
    x_conv_out = torch.zeros(64, 20, 128)
    higway = Highway(e_word)
    x_highway = higway(x_conv_out)
    if torch.sum(x_conv_out - x_highway).item() == 0:
        print("zero edge case check pass")


