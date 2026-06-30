"""
Transformer Encoder (BERT-style) — from scratch.

Core components:
  1. Embedding: token + position + segment (converts IDs → dense vectors)
  2. Multi-Head Self-Attention:  ←  "semantic aggregation"
  3. EncoderBlock: MHA + FFN + LayerNorm + Residual
  4. MLMHead: predict masked tokens  ←  "entropy increase noise reduction"
  5. ClassificationHead: [CLS] → sentiment

The attention mechanism is the key innovation of the Transformer.
Each token can "attend" to every other token, weighting their influence
by learned relevance. This enables rich contextual representation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
#  Positional Encoding
# ─────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).

    Adds position information to token embeddings so the model knows
    where each token sits in the sequence. We use fixed sinusoids
    instead of learned embeddings so the model can generalise to
    sequence lengths not seen during training.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


# ─────────────────────────────────────────────────────────
#  Multi-Head Self-Attention  ←  "semantic aggregation"
# ─────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention.

    The core idea: each token projects itself into three spaces:
      Q (Query):   "what am I looking for?"
      K (Key):     "what do I contain?"
      V (Value):   "what information do I carry?"

    Attention weights = softmax(Q @ K^T / sqrt(d_k))
    Output = Attention @ V

    Multiple heads let the model attend to different relationship types
    simultaneously (e.g., one head for syntax, another for semantics).

    This is the implementation of "semantic aggregation": each token's
    final representation is a weighted sum of ALL tokens in the sequence,
    where weights are learned based on relevance.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns: (batch, seq_len, d_model)
        Also returns attention weights for visualisation.
        """
        B, T, D = x.shape

        # Project to Q, K, V, then split into heads.
        # Shape: (batch, n_heads, seq_len, d_k)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # Apply padding mask: set attention to -inf for padding positions.
        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))

        # Softmax over the last dimension (attention over sequence).
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values.
        out = attn @ V  # (B, n_heads, T, d_k)
        # Concatenate heads.
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        # Final projection.
        out = self.W_o(out)
        return out, attn


# ─────────────────────────────────────────────────────────
#  Feed-Forward Network
# ─────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation.

    Why GELU over ReLU? GELU is a smooth version of ReLU that has been
    shown to work better in Transformers (used in BERT/GPT).

    FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2

    The inner dimension is typically 4× d_model (e.g., 512 → 2048 → 512).
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ─────────────────────────────────────────────────────────
#  Encoder Block
# ─────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """
    One Transformer encoder block:

      x → Multi-Head Attention → Add + LayerNorm → FFN → Add + LayerNorm

    Residual connections (Add) let gradients flow directly through the block,
    enabling training of deep Transformers. LayerNorm stabilises activations
    by normalising across the feature dimension.

    Each block applies "semantic aggregation" (via attention) followed by
    "semantic composition" (via FFN).
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual.
        attn_out, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN + residual.
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, attn_weights


# ─────────────────────────────────────────────────────────
#  Full BERT model
# ─────────────────────────────────────────────────────────

class BERT(nn.Module):
    """
    BERT (Bidirectional Encoder Representations from Transformers).

    A stack of Transformer encoder blocks with token+position+segment
    embeddings. Designed for pre-training via Masked Language Model (MLM)
    and fine-tuning on downstream tasks.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Token embedding: maps token IDs to dense vectors.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # Segment embedding: distinguishes sentence A vs B (not used here, kept for BERT compat).
        self.segment_embed = nn.Embedding(2, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        # Stack of encoder blocks.
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # LayerNorm at the output.
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        input_ids: (batch, seq_len) token indices
        attention_mask: (batch, seq_len) 1=real, 0=padding
        token_type_ids: (batch, seq_len) 0=sentence A, 1=sentence B

        Returns: (batch, seq_len, d_model) + list of attention matrices
        """
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        seq_len = input_ids.size(1)

        # Sum token + segment embeddings.
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)
        x = x + self.segment_embed(token_type_ids)

        # Add positional encoding.
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through encoder blocks.
        all_attentions = []
        for block in self.blocks:
            x, attn = block(x, attention_mask)
            all_attentions.append(attn)

        x = self.norm(x)
        return x, all_attentions


# ─────────────────────────────────────────────────────────
#  MLM Head  ←  "entropy increase noise reduction"
# ─────────────────────────────────────────────────────────

class MLMHead(nn.Module):
    """
    Masked Language Model head.

    Predicts the original token at masked positions.
    This is the "denoising" step that reverses the "entropy increase"
    caused by random masking.

    Logic: randomly mask 15% of tokens, train the model to predict them.
      - 80% of masked: replaced with [MASK]
      - 10% of masked: replaced with random token
      - 10% of masked: unchanged
    This prevents the model from relying solely on [MASK] and forces
    it to build robust contextual representations.

    The loss is computed only on masked positions → the model learns
    by "denoising" the corrupted input.
    """

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        """hidden_states: (batch, seq_len, d_model) → (batch, seq_len, vocab)"""
        x = F.gelu(self.dense(hidden_states))
        x = self.norm(x)
        return self.decoder(x)


# ─────────────────────────────────────────────────────────
#  Classification Head (for downstream fine-tuning)
# ─────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    Simple classifier on top of [CLS] token.

    After pre-training, the [CLS] token's representation captures the
    aggregate meaning of the entire sequence. We fine-tune by adding
    a small classifier on top.
    """

    def __init__(self, d_model, num_classes=2, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_token):
        """cls_token: (batch, d_model) → (batch, num_classes)"""
        return self.fc(self.dropout(cls_token))


# ─────────────────────────────────────────────────────────
#  Convenience factory
# ─────────────────────────────────────────────────────────

class BERTForMLM(nn.Module):
    """BERT + MLM head for pre-training."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, max_len=128, dropout=0.1):
        super().__init__()
        self.bert = BERT(vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout)
        self.mlm = MLMHead(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden, attentions = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.mlm(hidden)
        return logits, attentions

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class BERTForClassification(nn.Module):
    """BERT + classification head for fine-tuning."""

    def __init__(self, vocab_size, num_classes=2, d_model=128, n_heads=4,
                 n_layers=4, d_ff=None, max_len=128, dropout=0.1):
        super().__init__()
        self.bert = BERT(vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout)
        self.classifier = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        hidden, attentions = self.bert(input_ids, token_type_ids, attention_mask)
        # Use [CLS] token (first position).
        cls_out = hidden[:, 0, :]
        logits = self.classifier(cls_out)
        return logits, attentions

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
