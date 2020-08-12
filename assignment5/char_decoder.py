#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""


import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        char_embeds = self.decoderCharEmb(input)
        if dec_hidden is not None:
            output, (h_n, c_n) = self.charDecoder(char_embeds, dec_hidden)
        else:
            output, (h_n, c_n) = self.charDecoder(char_embeds)
        
        scores = self.char_output_projection(output)

        return scores, (h_n, c_n)
        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        pad_idx = self.target_vocab.char_pad
        inputs = char_sequence[0:-1, :]
        targets = char_sequence[1:, :]

        scores, _ = self.forward(inputs, dec_hidden)
        scores = torch.flatten(scores, start_dim = 0, end_dim = 1)
        targets = torch.flatten(targets)

        """
        zero_loss_index = (targets == pad_idx).nonzero().squeeze(-1)

        loss = nn.CrossEntropyLoss(reduction = 'none')(scores, targets)
        #loss[[zero_loss_index]] = 0
        loss = torch.sum(loss)
        """
        loss = nn.CrossEntropyLoss(ignore_index = pad_idx, reduction = 'sum')(scores, targets)
        return loss
        ### END YOUR CODE
        
    
    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        h_0, c_0 = initialStates
        _, batch_size, hidden_size = h_0.size()
        current_char_idxs = torch.ones(1, batch_size, dtype = torch.long, device = device) * self.target_vocab.start_of_word

        output_word = torch.empty(0, batch_size, dtype=torch.long , device=device)
        for i in range(max_length):
            scores, (h_0, c_0) = self.forward(current_char_idxs, (h_0, c_0))
            p = nn.Softmax(dim = -1)(scores)
            current_char_idxs = torch.argmax(p, dim = -1)
            output_word = torch.cat((output_word, current_char_idxs), dim=0)
        
        decodedWords = []
        for b in range(batch_size):
            char_idx_list = output_word[:, b].tolist()
            chars_list = [self.target_vocab.id2char[i] for i in char_idx_list]
            w = ''
            for c in chars_list:
                if c != "{" and c != "}":
                    w += c
                else:
                    break
            decodedWords.append(w)

        return decodedWords

        ### END YOUR CODE
