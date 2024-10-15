import torch
import torch.nn as nn
from utils import clones, LayerNorm, SublayerConnection, PositionwiseFeedForward
from attention import MultiHeadedAttention

# --- [3] DecoderLayer and Decoder
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size: int, self_attn: MultiHeadedAttention, src_attn: MultiHeadedAttention, feed_forward: PositionwiseFeedForward, dropout: float):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        
        # print(f"[DECODER] Decoder target: {x.size()}")
        # print(f"[DECODER] Decoder memory: {memory.size()}")
        # print(f"[DECODER] Decoder input mask: {src_mask.size()}")
        # print(f"[DECODER] Decoder target mask: {tgt_mask.size()}")
        
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x = self.norm(x)
        
        # print(f"[DECODER] Decoder output: {x.size()}")
        
        return x