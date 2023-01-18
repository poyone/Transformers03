import torch
import torch.nn as nn
import math
from utils import clones

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_head, QKV_O_linear = 4, drop_rate=0.1):
        super().__init__()
        assert d_model % n_head == 0

        self.d_model = d_model
        self.n_head = n_head

        self.dropout = nn.Dropout(p=drop_rate)
        self.linear = clones(nn.Linear(d_model, d_model) ,QKV_O_linear)
    
    def forward(self, Q, K, V, mask=None):
        batch_size, _, emb_dim = Q.shape    # (batch_size, seq_len, emb_dim)
        d_k = emb_dim // self.n_head

        Q_heads = self.linear[0](Q).view(batch_size, self.n_head, -1, d_k).transpose(0,1)
        K_heads = self.linear[1](K).view(batch_size, self.n_head, -1, d_k).transpose(0,1)
        V_heads = self.linear[2](V).view(batch_size, self.n_head, -1, d_k).transpose(0,1)

        V_att, scores_att= ScaledAttention(Q_heads, K_heads, V_heads, d_k,  mask, self.dropout)
        V_att = V_att.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_head * d_k)
        V_att = self.linear[3](V_att)
        K_lin = K_heads.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_head * d_k)

        return  K_lin, V_att, # scores_att


def ScaledAttention(query, key, value, d_k, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch_size, channel, head, d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = scores.softmax(dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    value = torch.matmul(scores, value)

    return value, scores



if __name__ == '__main__':
    source = torch.randn(4, 1, 28, 28)
    mul = MultiheadAttention(784, 7)
    mul(source)
