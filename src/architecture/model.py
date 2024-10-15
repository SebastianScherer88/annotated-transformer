import torch.nn as nn
from utils import PositionwiseFeedForward, copy
from attention import MultiHeadedAttention
from embedding import PositionalEncoding, Embeddings
from encoder import Encoder, EncoderLayer
from decoder import Decoder, DecoderLayer, subsequent_mask
from generator import Generator
import torch

# --- [5] EncoderDecorder Model

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: nn.Sequential, tgt_embed: nn.Sequential, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        print(f"Target: {type(tgt)}")
        print(f"Target mask: {type(tgt_mask)}")
        encoder_out = self.encode(src, src_mask)
        print(f"Encoder output: {type(encoder_out)}")
        decoder_out = self.decode(encoder_out, src_mask, tgt, tgt_mask)
        print(f"Decoder output: {type(decoder_out)}")
        return decoder_out

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
def make_model(
    src_vocab_size: int, tgt_vocab_size: int, N: int=6, d_model: int=512, d_ff: int=2048, h:int=8, dropout: float=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
        Generator(d_model, tgt_vocab_size),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def test_model(
    src_vocab_size: int=5, tgt_vocab_size: int=6, N: int=1, d_model: int=4, d_ff: int=8, h:int=2, dropout: float=0.1, n_out: int=10
    ):
    test_model = make_model(src_vocab_size,tgt_vocab_size,N,d_model,d_ff,h,dropout)
    test_model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4]])
    src_mask = torch.ones(1, 1, 5)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)
    
    print(f"[TEST] Source : {src} (size {src.size()})")
    print(f"[TEST] Source mask: {src_mask} (size {src_mask.size()})")
    print(f"[TEST] Memory: {memory} (size {memory.size()})")
    print(f"[TEST] Target: {ys} (size {ys.size()})")
    print("[TEST] Starting inference...")

    for i in range(n_out-1):
        print(f"[TEST] Predicting output token {i}.")
        ys_mask = subsequent_mask(ys.size(1)).type_as(src.data)
        print(f"[TEST] Decoder target mask for token {i}: {ys_mask} (size {ys_mask.size()})")
        out = test_model.decode(
            memory, src_mask, ys, ys_mask
        )
        print(f"[TEST] Decoder output for token {i}: {out} (size {out.size()})")
        generator_in = out[:, -1]
        print(f"[TEST] Generator input for token {i}: {generator_in} (size {generator_in.size()})")
        prob = test_model.generator(generator_in)
        print(f"[TEST] Generator output for token {i}: {prob} (size {prob.size()})")
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
        print(f"[TEST] Decoder target for token {i}: {ys} (size {ys.size()})")
        print("==============================")

    print("Example Untrained Model Prediction:", ys)
    
def run_model_test(n_test: int = 10, n_out: int = 10):
    for _ in range(n_test):
        test_model(n_out=n_out)