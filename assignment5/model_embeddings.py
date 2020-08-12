#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.e_char = 50
        self.char_embeddings = nn.Embedding(len(self.vocab.char2id), self.e_char,
                                                padding_idx = vocab.char_pad)

        self.cnn = CNN(self.word_embed_size, self.e_char)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        char_embeds = self.char_embeddings(input)
        char_embeds = char_embeds.transpose(2, 3)

        x_conv_out = self.cnn(char_embeds)
        x_highway = self.highway(x_conv_out)
        output = self.dropout(x_highway)

        return output
        ### END YOUR CODE

if __name__ == '__main__':
    import torch

    print("model embeddings sanity check!")
    class dummy_vocab(object):
        def __init__(self):
            self.char2id = [0, 1, 2, 3, 4, 5, 6, 7, 8,9,10, 11, 12,13,14,15]
    
    vocab = dummy_vocab()
    embedding_layer = ModelEmbeddings(512, vocab)
    input = torch.randint(0, 15, (15, 64, 13), dtype = torch.long)
    output = embedding_layer(input)

    if output.shape == (15, 64, 512):
        print("Pass the output shape check")

