import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp.bert.model import EncoderBlock, PositionalEncoding


class CrossAttention(nn.Module):
    """Multi-head attention where Q from decoder, K,V from encoder."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, encoder_mask=None):
        B, T, D = x.shape
        S = encoder_output.size(1)

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_output).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_output).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask[:, None, None, :S] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class DecoderBlock(nn.Module):
    """Decoder block: causal self-attn → cross-attn → FFN."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_mask=None, encoder_mask=None):
        # Causal self-attention.
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention: Q from decoder, K,V from encoder.
        cross_out = self.cross_attn(x, encoder_output, encoder_mask)
        x = self.norm2(x + self.dropout(cross_out))

        # FFN.
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Transformer(nn.Module):
    """Full Seq2Seq Transformer: Encoder → Decoder."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 d_ff=None, max_len=128, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # Shared embeddings (weight-tying between encoder, decoder, and LM head).
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight-tying: LM head shares weights with embedding.
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _causal_mask(self, size, device):
        """Upper-triangular mask for causal self-attention."""
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def encode(self, src_tokens, src_mask=None):
        """Encode source sequence → encoder hidden states."""
        x = self.embed(src_tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for block in self.encoder_blocks:
            x, _ = block(x, src_mask)
        return self.norm(x)

    def decode(self, tgt_tokens, encoder_output, src_mask=None):
        """Decode target sequence from encoder hidden states."""
        T = tgt_tokens.size(1)
        causal_mask = self._causal_mask(T, tgt_tokens.device)

        x = self.embed(tgt_tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.dropout(x)

        for block in self.decoder_blocks:
            x = block(x, encoder_output, self_mask=causal_mask, encoder_mask=src_mask)
        return self.norm(x)

    def forward(self, src_tokens, tgt_tokens, src_mask=None):
        """Full forward pass: encode → decode → logits."""
        encoder_output = self.encode(src_tokens, src_mask)
        decoder_output = self.decode(tgt_tokens, encoder_output, src_mask)
        return self.lm_head(decoder_output)

    def generate(self, src_tokens, src_mask=None, max_len=None, bos_idx=2, eos_idx=3):
        """Greedy decoding: generate target sequence one token at a time."""
        self.eval()
        device = src_tokens.device
        batch_size = src_tokens.size(0)

        max_len = max_len or (self.pos_enc.pe.size(1) - 1)

        with torch.no_grad():
            encoder_output = self.encode(src_tokens, src_mask)

            tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

            for _ in range(max_len):
                logits = self.lm_head(self.decode(tgt, encoder_output, src_mask))
                next_token = logits[:, -1:].argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)

                finished = finished | (next_token.squeeze(1) == eos_idx)
                if finished.all():
                    break

        return tgt

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
