import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_V, d_K, h):
        self.d_model = d_model
        self.d_V = d_V
        self.d_K = d_K
        self.d_Q = self.d_K
        self.h = h

        self.Weight_Q = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h, d_model, self.d_Q)))
        self.Weight_V = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h, d_model, self.d_V)))
        self.Weight_K = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h, d_model, self.d_K)))

        self.Weight_O = nn.Parameter(nn.init.xavier_uniform_(torch.empty(h*d_V, d_model)))

    
    def forward(self, V, K, Q, mask = None):
        """
        @param V: input value, tensor with shape (sentence_length, batch, d_model)
        @param K: input key, tensor with shape (sentence_length, batch, d_model)
        @param Q: input query, tensor with shape (sentence_length, batch, d_model)
        @mask: Used for the transformer of decoder, creating causality (predicting condition on previous word)
        """
        s_len, b, _ = V.size()

        assert V.size(0) == K.size(0) == Q.size(0), "Unmatched sentence length in K, V and Q"
        assert V.size(1) == K.size(1) == Q.size(1), "Unmatched batch size in K V and Q"

        #Do the linear projection over input K, V and Q
        V = torch.flatten(V, 0, 1)
        V_proj = V @ torch.flatten(torch.transpose(self.Weight_V, 0, 1), 1, 2)
        V_proj = V_proj.view(s_len * b, self.h, self.d_Q)
        #V_proj = V_proj.view(s_len, b, self.h, self.d_Q)

        K = torch.flatten(K, 0, 1)
        K_proj = K @ torch.flatten(torch.transpose(self.Weight_K, 0,1), 1, 2)
        K_proj = K_proj.view(s_len * b, self.h, self.d_K)
        #V_proj = K_proj.view(s_len, b, self.h, self.d_K)

        Q = torch.flatten(Q, 0, 1)
        Q_proj = Q @ torch.flatten(torch.transpose(self.Weight_Q, 0,1), 1, 2)
        Q_proj = Q_proj.view(s_len * b, self.h, self.d_Q)
        #Q_proj = K_proj.view(s_len, b, self.h, self.d_Q)

        #Calculate the attention score
        atten = torch.flatten(Q_proj, 0, 1) @ torch.transpose(torch.flatten(K_proj, 0, 1), 0, 1)
        




