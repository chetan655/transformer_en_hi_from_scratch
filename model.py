import torch
import torch.nn as nn
import math
 
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)    # this creates a lookup table, a matrix of size (vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

    # output from this class will be -> (batch_size, seq_len, d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)  # creating a matrix of dim, (seq_len, d_model)

        # creating a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)  # unsqueeze(1) turns shape [n] into [n, 1] by adding a dimension at index 1.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/ d_model))

        # sin for even & cos for odd positions
        pe[:, 0::2] = torch.sin(position * div_term)   # pe[:, 0::2] means All positions (rows), and only even-numbered dimensions (columns 0, 2, 4, ...).
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Adding a (1, seq_len, d_model) tensor to (batch_size, seq_len, d_model) works because PyTorch broadcasts over the batch dimension.

        self.register_buffer('pe', pe)  # it saves (pe) inside the module(self) as 'pe' we need positional encoding to be saved. but it is not learnable parameter so we save it like this. it will be saved in state_dict

        # so here pe is of shape -> (1, seq_len, d_model)

    def forward(self, x):
        # adding word embedding and positional embedding
        '''
            x -> (batch_size, seq_len, d_model)
            x.shape[1] -> seq_len
            [:, :x.shape[1], :] -> [all the batches, dimension from pe upto seq_len, all the d_model]
        '''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)  # taking all the rows
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))   # it will multiply
        self.bias = nn.Parameter(torch.zeros(1))   # it will be added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)   # -1 means it will calculate mean across 512 features (batch_size, seq_len, embedding(512))
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x = self.linear_1(x)
        # x = torch.relu(x)
        # x = self.dropout(x)
        # x = self.linear_2(x)
        # return x

        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"   # assert condition, statement ( condition fails)  -> if condition fails it raise assertionError
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)   # query weight matrix
        self.w_k = nn.Linear(d_model, d_model)   # key weight matrix
        self.w_v = nn.Linear(d_model, d_model)   # value weight matrix
        self.w_o = nn.Linear(d_model, d_model)   # matrix at the end used for linear transformation
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # [batch, h, seq_len, d_k] -> [batch, h, seq_len, seq_len]    # every attention head gets some part of the sentence
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)   
        if mask is not None:
            attention_score.masked_fill_(mask == 0, -1e9)            # it will replace values that are 0 to -1e9
        attention_score = attention_score.softmax(dim = -1)  # [batch, h, seq_len, seq_len]
        
        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score
 
    def forward(self, q, k, v, mask):           # if we do not want some words to interect with some other then we mask them
        query = self.w_q(q)   # (batch, seq_len, d_model) -> (batch, seq_len, d_model) and same for below three also
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)    # [batch, seq_len, d_model] -> [batch, seq_len, num_head, d_k] -> [batch, num_head, seq_len, d_k]
        key = key.view(key.shape[0], key.shape[1], self.h, self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # [batch, h, seq_len, d_k] -> [batch, seq_len, h, d_k] -> [batch, seq_len, d_model]
        x = x.transpose(1,2).contigious().view(x.shape[0], -1, self.h * self.d_k)

        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        return self.w_o(x)
    

    '''
        MultiHeadAttention:-
        - suppose we have 2 tokens and their embedding is of 512 dim. so we stack them to send in. Now their shape is (2x512)
        - these embedding will make 3 matrix named (query, key, value)
        - input (2x512) will be multiplied with 8 * (query_weight_matrix, key_weight_matrix, value_weight_matrix), each of shape (512x512)
        - output from above calculation is 8 * (query_matrix, key_matrix, value_matrix), each of shape (2x512)
        - we reshape it from (2x512), assume batch_size = 1, so we reshape it from (1x2x512) -> (1x2x8x64) or we can say [(batch_size, seq_len, d_model) -> (batch_size, seq_len, heads(h), each_head_dim(d_model//h))]
        - (batch_size, seq_len, h, d_k) this line means for each token we have 8 heads of dim 64
        - we transpose it to (batch_size, h, seq_len, d_k), now this line means (for each head we have all tokens of dim 64)
        - before transposing each token is getting all heads, after transposing each head getting all tokens some part.
        - now all query, key, value matrix have shape (batch_size, h, seq_len, d_k)
        - they will be sent to attention function
        - to calculate attention score we multiple (query * key.T) means, [(batch_size, h, seq_len, d_k)*(batch_size, h, seq_len, d_k).transpose(-2, -1)] -> [(batch_size, h, seq_len, d_k) * (batch_size, h, d_k, seq_len)] * sqrt(d_k)
        - now we have attention score shape as -> [batch_size, h, seq_len, seq_len] 
        - x = attention_score * value
        - [(batch_size, h, seq_len, seq_len) * (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, d_k)]
        - now transpose x as x.transpose(1,2) -> (batch_size, seq_len, h, d_k)
        - x.transpose changes shape, but memory is not contiguous
        - .contigious makes a new, properly ordered block of memory
        - .view now safely reshapes it
        - after applying .view we get original matrix [batch_size, seq_len, d_model]
    '''
        

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))    # sublayer can be ffnn or multi-head attention and x is the raw input
        # self.norm(x + self.dropout(sublayer(x)))  # use in gpt/ bert


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))   # we have multiple arguments in forward method of multi-head attention so we use anonymous function lambda
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)