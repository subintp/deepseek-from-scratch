
import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout,num_heads,qkv_bias=False):
        super().__init__()
        print("multi headed attention initialized")
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_in = d_in
        self.d_out = d_out

        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_out = nn.Linear(d_out, d_out) # linear layer to combine the head outputs
        self.dropout = nn.Dropout(dropout)
        

    def forward(self,x):
        print("multi headed attention forward")
        
        # batch_size, num_tokens, input dimension 
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        print("Q:", queries, "K:", keys, "V:", values)

        # Implicitly split  the matrix by adding  a num_heads dimension 
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        print("Reshaped Q:", queries, "Reshaped K:", keys, "Reshaped V:", values)

        # Transpose: (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)
        print("Transposed Q:", queries, "Transposed K:", keys, "Transposed V:", values)
        
        # Scaled dot product of attention with causual mask
        attention_scores = queries @ keys.transpose(2, 3) # dot product for each head
        mask = torch.triu(torch.ones_like(attention_scores), diagonal=1) # causual mask
        attention_scores = attention_scores.masked_fill(mask == 1, -torch.inf) # mask the future tokens
        print("attention_scores:", attention_scores)

        # softmax
        attention_weights = torch.softmax(attention_scores / (keys.shape[-1]**0.5), dim=-1)
        print("attention_weights:", attention_weights)

        # dropout
        attention_weights = self.dropout(attention_weights)
        print("attention_weights:", attention_weights)


        # dot product and transpose from (b, num_heads, num_tokens, head_dim) to (b, num_tokens, num_heads, head_dim)
        context_vector = (attention_weights @ values).transpose(1, 2)
        
        # merge context vector
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
    
        return context_vector
        
        
