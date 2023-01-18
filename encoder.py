import torch
import torch.nn as nn
from multihead_attention import MultiheadAttention 
from utils import clones

'''
encoder_layer的流程为qkv送入Multihead_Attention
将得出的kv进行add_norm
最终输送给decoder
'''
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_rate=0.1):
        super().__init__()
        self.mul = MultiheadAttention(d_model, n_head, drop_rate=drop_rate)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential( nn.Linear(d_model, d_model*4),
                                 nn.ReLU(),
                                 nn.Linear(d_model*4, d_model)) 

    def forward(self, Q, K, V, MASK):
        K, V = self.mul(Q, K, V, MASK)
        V_1 = self.norm(V + Q)
        K_1 = self.norm(K + Q)

        V_2= self.ff(V_1)
        K_2= self.ff(K_1)
        V_2 = self.norm(V_1 + V_2)
        K_2 = self.norm(K_1 + K_2)

        return K_2, V_2

'''
encoder主要是实现多个layer的堆叠传输
'''
class Encoder(nn.Module):
    def __init__(self, model, N):
        super().__init__()
        self.layers = clones(model, N)

    def forward(self, q,k,v, mask):
        for layer in self.layers:
            K, V = layer(q,k,v,mask)
        return K, V


if __name__ == '__main__':
    source = torch.randn(4, 2, 28)
    ecd_layer = EncoderLayer(784, 7)
    ecd = Encoder(ecd_layer, 2)
    ecd(source)