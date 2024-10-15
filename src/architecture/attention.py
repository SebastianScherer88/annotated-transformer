import torch
import torch.nn as nn
import math

from utils import clones

def attention(query, key, value, coefficient_mask = None, dropout=None):
    # Expected tensor dimensions:
    # query: # (n_batch, n_query, d_k)
    # key: # (n_batch, n_key, d_k)
    # value: # (n_batch, n_key, d_k)
    d_k = query.size(-1)
    softmax_input = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) # (n_batch, n_query, n_key)
    
    if coefficient_mask is not None:
        softmax_input = softmax_input.masked_fill(coefficient_mask == 0, -1e9) # (n_batch, n_query, n_key)
    
    coefficient = softmax_input.softmax(dim=-1) # (n_batch, n_query, n_key)
    
    if dropout is not None:
        coefficient = dropout(coefficient) # (n_batch, n_query, n_key)
        
    attention = torch.matmul(coefficient, value) # (n_batch, n_query, d_k)
    
    return attention, coefficient

class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout:float=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        d_k = d_model // h
        self.d_k = d_k
        self.h = h
        self.dropout = nn.Dropout(p=dropout)
        self.head_projections = [clones(nn.Linear(d_model, d_k), 3) for i in range(h)]
        self.final_projection = nn.Linear(d_model,d_model)
        
        # storage attribute for visualization purposes. is re-populated for each
        # forward pass to hold the coefficient return from the `attention` 
        # method. 
        # Expected dimension: (n_batch, n_sequence, n_sequence, h)
        self.attn = None
        
    def forward(self, query, key, value, mask=None):
        # Expected tensor dimensions:
        # query: # (n_batch, n_query, d_model)
        # key: # (n_batch, n_key, d_model)
        # value: # (n_batch, n_key, d_model)
        
        # initialize storage for each head's outputs
        head_attentions = []
        head_coefficients = []
        
        for W_q, W_k, W_v in self.head_projections:
            head_query_w = W_q(query) # (n_batch, n_query, d_k)
            head_key_w = W_k(key) # (n_batch, n_key, d_k)
            head_value_w = W_v(value) # (n_batch, n_key, d_k)
            
            head_attention = attention(head_query_w, head_key_w, head_value_w, mask, self.dropout) # (n_batch, n_query, d_k)
            head_attentions.append(head_attention[0])
            head_coefficients.append(head_attention[1])
            
        # combine individual heads' attention and coefficient outputs
        multi_attention = torch.concat(head_attentions,-1) # (n_batch, n_query, d_model)
        coefficient = torch.stack(head_coefficients, -1) # (n_batch, n_query, n_key, h)
        
        # populate coefficient storage attribute for this forward pass
        self.attn = coefficient # (n_batch, n_query, n_key, h)
        
        # apply final linear projection for this layer
        projected_attention = self.final_projection(multi_attention)
        
        return projected_attention