import torch
import torch.nn as nn
from multihead_attention import MultiheadAttention, clones
from utils import attn_mask

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_rate=0.1):
        super().__init__()
        self.mmul = MultiheadAttention(d_model, n_head, drop_rate=drop_rate)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential( nn.Linear(d_model, d_model*4),
                                 nn.ReLU(),
                                 nn.Linear(d_model*4, d_model)) 

    def forward(self, q, k, v, dec_mask, cro_mask):

        _, V = self.mmul(q, q, q, mask=dec_mask)
        Q_1 = self.norm(V + q)

        '''
        这里写V_2_是因为还要传入下一层,
        上面写Q是因为 decoder在上一步只提供Q与下一层的多头层计算
        '''
        _, V_2_ = self.mmul(Q_1, k, v, mask=cro_mask)
        V_2 = self.norm(V_2_ + Q_1)

        V_2= self.ff(V_2)
        V= self.norm(V_2 + V_2_) # 至此decoder的一轮计算完成

        return V

class Decoder(nn.Module):
    def __init__(self, model, N):
        super().__init__()
        self.layers = clones(model, N)

    def forward(self, x, k, v, dec_mask, cro_mask):
        for layer in self.layers:
            V = layer(x, k, v, dec_mask, cro_mask)
        return V

if __name__ == '__main__':
    x = torch.randn(4, 1, 28, 28)
    k = torch.randn(4, 1, 28, 28)
    v = torch.randn(4, 1, 28, 28)
    dcd_layer = DecoderLayer(784, 7)
    dcd = Decoder(dcd_layer, 2)
    dcd(x, k, v)