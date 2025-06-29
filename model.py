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

    def forward(self, x):
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