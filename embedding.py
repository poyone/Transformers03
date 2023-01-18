import torch.nn as nn
import torch
import math

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model ):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model) # 对embedding的值缩放 制position数值的影响


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model , dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # 所有行，列起始位为1，步长为2
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)    # 从(batch_size, seq_len) --> (1.)

        self.register_buffer('pe', pe)  # 设置缓冲区，表示参数不更新
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == '__main__':
    x = torch.randint(1,40, (2, 28))
    emb = Embeddings(40, 784)
    pe = PositionalEncoding(40, 784)
    print((pe(emb(x))).shape)