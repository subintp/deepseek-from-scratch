
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        print("self attention initialized")
        self.d_in = d_in
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        

    def forward(self,x):
        print("self attention forward")
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        attention_scores = queries @ keys.T
        mask = torch.triu(torch.ones_like(attention_scores), diagonal=1)
        attention_scores = attention_scores.masked_fill(mask == 1, -torch.inf)
        attention_weights = torch.softmax(attention_scores/torch.sqrt(torch.tensor(self.d_in)), dim = -1)
        context_vector = attention_weights @ values
        return context_vector
        
        
