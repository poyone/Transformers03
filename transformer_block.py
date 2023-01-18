import torch
import torch.nn as nn
from utils import attn_mask
from embedding import Embeddings, PositionalEncoding
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer

class Transformer(nn.Module):
    def __init__(self,enc_vocab_size, dec_vocab_size, d_model, n_head, num_layer):
        super(Transformer,self).__init__()

        self.encoder_emb = Embeddings(enc_vocab_size, d_model)
        self.decoder_emb = Embeddings(dec_vocab_size, d_model)
        self.encoder_pos = PositionalEncoding(enc_vocab_size, d_model)
        self.decoder_pos = PositionalEncoding(dec_vocab_size, d_model)

        self.encoder = Encoder(EncoderLayer(d_model=d_model,n_head= n_head), N=num_layer)
        self.decoder = Decoder(DecoderLayer(d_model=d_model,n_head= n_head), N=num_layer)
        self.answer = nn.Linear(d_model, dec_vocab_size)

    def forward(self, src_q, tgt_q):
        # 生成掩码
        encoder_mask = attn_mask(src_ids=src_q)
        decoder_mask = attn_mask(tgt_ids=tgt_q)
        cross_mask = attn_mask(src_q, tgt_q)

        # encoder部分   一种mask
        src_q = self.encoder_emb(src_q) 
        src_q = self.encoder_pos(src_q)
        k, v = self.encoder(src_q, src_q, src_q, encoder_mask)

        # decoder部分   要传入两种mask
        tgt_q = self.decoder_emb(tgt_q)
        tgt_q = self.decoder_pos(tgt_q)
        output = self.decoder(tgt_q, k, v, decoder_mask, cross_mask)

        # 转换到tgt_vocab 做个argmax进行输出
        output = self.answer(output)

        return output


if __name__ == '__main__':
    x = torch.randint(1,40, (4, 23))
    y = torch.randint(1,40, (4, 17))
    trs = Transformer(40, 41, 784, 7, 2)
    output = trs(x, y)
    print(output.argmax(-1).shape)

