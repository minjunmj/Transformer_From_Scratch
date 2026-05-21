import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout:float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # softmax(QKT/sq(d_K))V  Q (B, h, N, d_k)
    def forward(self, Q, K, V, attn_mask=None):
        d_k = Q.size(-1)
        att = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(d_k)
        #(B, h, N, N)
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, -1e9)
        att_score = F.softmax(att, dim=-1)
        out = torch.matmul(att_score, V)

        return out
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.num_head = num_head
        self.d_k = d_model//num_head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def split_head(self, x):
        # x = (B, N, d_model)
        B, N, _ = x.size()
        x = x.view(B, N, self.num_head, self.d_k)
        x = x.permute(0, 2, 1, 3)
        return x

    def concate_head(self, x):
        # x = (B, h , N, d_k)
        B, h, N, d_k = x.size()
        x = x.permute(0, 2, 1, 3)
        x = x.contiguous()
        x = x.view(B, N, h*d_k)

        return x
    
    def forward(self, query, key, value, attn_mask=None):
   
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_head(Q)
        K = self.split_head(K)
        V = self.split_head(V)

        out = self.attn(Q, K, V, attn_mask=attn_mask)

        out = self.concate_head(out)
        out = self.W_o(out)

        return out

class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x-mean) / torch.sqrt(var + self.eps)
    
        return self.gamma * x + self.beta

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
    
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, num_head: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(num_head=num_head, d_model=d_model, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = LayerNorm(d_model=d_model)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

#att -> add+norm
    def forward(self, x, src_mask=None):
        out = self.attn(x, x, x, attn_mask = src_mask)
        out = self.dropout1(out)
        x = x + out
        x = self.norm1(x)
        ff_out = self.ffn(x)
        ff_out = self.dropout2(ff_out)
        x = x + ff_out
        x = self.norm2(x)

        return x
        
class Encoder(nn.Module):
    def __init__(self, num_layer: int, d_model: int, num_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(num_head, d_model, d_ff, dropout) for _ in range(num_layer)
        ])
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, num_head: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(num_head = num_head, d_model=d_model, dropout=dropout)
        self.cross_attn = MultiHeadAttention(num_head = num_head, d_model=d_model, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = LayerNorm(d_model=d_model)
        self.norm2 = LayerNorm(d_model=d_model)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

#mask att -> add+norm / cross att -> add + norm
    def forward(self, x, en, src_mask=None, tgt_mask=None):
        out = self.attn(x, x, x, attn_mask=tgt_mask)
        out = self.dropout1(out)
        x = x + out
        x = self.norm1(x)
        cross_out = self.cross_attn(x, en, en, attn_mask=src_mask)
        cross_out = self.dropout2(cross_out)
        x = x + cross_out
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        ffn_out = self.dropout3(ffn_out)
        x = x + ffn_out
        x = self.norm3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, num_layer: int, d_model: int, num_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(num_head, d_model, d_ff, dropout) for _ in range(num_layer)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, en, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, en, src_mask=src_mask, tgt_mask=tgt_mask)
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_len: int, num_layer: int, d_model: int, num_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(num_layer, d_model, num_head, d_ff, dropout)
     
        self.decoder = Decoder(num_layer, d_model, num_head, d_ff, dropout)

        self.proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def encode(self, src_ids, src_mask=None):
        en_embed = self.src_embed(src_ids)
        x = self.position(en_embed)
        
        return self.encoder(x, src_mask)
    
    def decode(self, tgt_ids, en, src_mask=None, tgt_mask=None):
        de_embed = self.tgt_embed(tgt_ids)
        x = self.position(de_embed)

        return self.decoder(x, en, src_mask, tgt_mask)
    
    def forward(self, src_ids, tgt_ids, src_mask, tgt_mask):
        en = self.encode(src_ids, src_mask)
        out = self.decode(tgt_ids, en, src_mask, tgt_mask)
        out = self.proj(out)

        return out