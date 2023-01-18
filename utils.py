import torch
import copy
import torch.nn as nn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attn_mask(src_ids=None, tgt_ids=None):
    # encoder的掩码 正方形
    if tgt_ids == None:
        mask = torch.matmul(src_ids.unsqueeze(-1), src_ids.unsqueeze(1))
        return mask

    # decoder第一层的掩码，返回的是一个上三角为1的矩阵(对角线上的没有操作为1)\
    # 因为统一是mask==0 进行填充所以这里反一下 返回的地方让 上三角=false，\
    # 这样就等于0 将在self-attention填充，正方形
    elif src_ids == None:
        mask = torch.triu(torch.ones((tgt_ids.shape[-1], tgt_ids.shape[-1])), diagonal=1).type(torch.uint8)
        return mask == 0

    # decoder第二层cro_mask的掩码，kv来自encoder，q和kv的形状可能不同，长方形
    else:
        Q= tgt_ids
        KV = src_ids
        mask = torch.matmul(Q.unsqueeze(-1), KV.unsqueeze(1))
    return mask