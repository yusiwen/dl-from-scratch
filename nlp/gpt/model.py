"""
GPT — Decoder-only Transformer with Causal Self-Attention and KV Cache.

Unlike BERT's bidirectional attention, GPT uses causal (masked) attention:
each token can only attend to itself and tokens before it. This enables
autoregressive generation: predict one token at a time, feed it back as
input, repeat.

Key components:
  1. CausalAttention: same as BERT's attention but with an upper-triangular
     mask that blocks "future" positions. During generation, we use a KV
     cache to avoid recomputing K and V for past tokens on every step.
  2. DecoderBlock: identical to BERT's EncoderBlock (MHA + FFN + residual).
     The only structural difference is the attention mask.
  3. LMHead: predicts the next token from the final hidden state (like BERT's
     MLM head, but applied to EVERY position instead of only [MASK]ed ones).

KV Cache:
  During training, we process the full sequence at once. During generation,
  we process one token at a time:
    - Step t: only compute Q_t, K_t, V_t for the new token.
    - Then K = concat(K_cache, K_t), V = concat(V_cache, V_t).
    - Attention uses the full K, V but only Q_t.
  This reduces per-step complexity from O(seq_len^2) to O(seq_len).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
#  Causal Self-Attention  (with optional KV Cache)
# ─────────────────────────────────────────────────────────

class CausalAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Same Q/K/V projection as BERT's attention, but with a causal mask:
    position i can only attend to positions j where j ≤ i.

    During generation, `use_cache=True` enables KV caching:
      - On the first call with a full sequence, K and V are cached.
      - On subsequent calls with a single token, only the new K, V are
        computed and appended to the cache. Q has only one row (the
        current token's query), so attention is O(seq_len) instead of
        O(seq_len^2).
    """

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

        # Causal mask: upper triangular (i attends to j ≤ i).
        # We register it as a buffer so it moves to the correct device.
        # shape: (1, 1, max_seq_len, max_seq_len) — broadcastable over
        # batch and head dimensions.
        self.register_buffer("causal_mask", None)

    def _build_causal_mask(self, seq_len, device):
        """
        Create an upper-triangular mask that prevents attending to future tokens.

        For seq_len=4:
          [[0, -inf, -inf, -inf],
           [0,    0, -inf, -inf],
           [0,    0,    0, -inf],
           [0,    0,    0,    0]]

        Positions with -inf will have softmax = 0 after the softmax.
        """
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        """
        x: (batch, seq_len, d_model)
        mask: optional (batch, 1, 1, seq_len) — padding mask
        use_cache: if True, enable KV caching
        past_kv: optional (past_k, past_v) from previous steps

        Returns: (output, present_kv)
          output: (batch, seq_len, d_model)
          present_kv: (k, v) tuple for caching
        """
        B, T, D = x.shape

        # Q, K, V projections.
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ---- KV Cache ----
        # During generation, we process one token at a time.
        # The first call processes the full prompt and caches K, V.
        # Subsequent calls only process the new token.
        #
        # Without cache: each new token requires recomputing attention
        # over ALL past tokens (O(T^2) per step).
        # With cache: only need O(T) per step (one new Q, append to K/V).
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv
            # Concatenate past K, V with new K, V.
            K = torch.cat([past_k, K], dim=1)
            V = torch.cat([past_v, V], dim=1)

        present_kv = (K, V) if use_cache else None

        # Reshape to multi-head: (B, n_heads, T, d_k).
        new_T = K.size(1)  # This is the total sequence length (past + new).
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, new_T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, new_T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention.
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, n_heads, T, new_T)

        # Apply causal mask (block future positions).
        causal_mask = self._build_causal_mask(new_T, scores.device)
        # If T < new_T (cached case), we need to slice the mask to (T, new_T).
        # T is the current query length, new_T is the total key length.
        scores = scores + causal_mask[:, :, :T, :new_T]

        # Apply padding mask (if provided).
        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T)
            scores = scores.masked_fill(mask[:, None, None, :T] == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V                               # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(out)

        return out, present_kv


# ─────────────────────────────────────────────────────────
#  Feed-Forward Network (same as BERT)
# ─────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Two-layer MLP with GELU activation and inner dimension 4× d_model.
    Same as the Encoder block's FFN.
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
#  Decoder Block
# ─────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    One GPT decoder block:

      x → Causal Self-Attention → Add + LayerNorm → FFN → Add + LayerNorm

    Structurally identical to BERT's EncoderBlock, but with causal attention
    and KV cache support. The difference is in the attention mask, not the
    block architecture.
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.attention = CausalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        """
        x: (batch, seq_len, d_model)
        past_kv: optional (k, v) tuple for this specific layer from previous steps
        """
        attn_out, present_kv = self.attention(x, mask, use_cache, past_kv)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, present_kv


# ─────────────────────────────────────────────────────────
#  GPT model (Decoder-only Transformer)
# ─────────────────────────────────────────────────────────

class GPT(nn.Module):
    """
    Decoder-only Transformer for autoregressive language modelling.

    Architecture overview:
      Token Embedding → Positional Encoding → N× DecoderBlock → LayerNorm → LM Head

    Training: given tokens [t1, t2, ..., tn], predict [t2, t3, ..., tn+1].
    Generation: start with a prompt, predict one token at a time, feed it back.
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=4,
                 d_ff=None, max_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Token embedding: maps each token ID to a dense vector.
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # Positional embedding: learnable (same as BERT, but GPT uses learned PE).
        # Why learned instead of sinusoidal? GPT uses learned embeddings that
        # can adapt to the data distribution. For small models it matters less.
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stack of decoder blocks.
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # LM head: predicts next-token logits.
        # Weight-typing is omitted here because shared tensor references
        # cause issues when loading state_dicts on MPS.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small normal values for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, use_cache=False, past_kv=None):
        """
        input_ids: (batch, seq_len) token indices
        attention_mask: (batch, seq_len) 1=real, 0=padding
        use_cache: enable KV cache (for generation)
        past_kv: list of (k, v) tuples, one per layer

        Returns:
          logits: (batch, seq_len, vocab_size)
          present_kv: list of (k, v) tuples, one per layer (for caching)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings.
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)

        # Positional embeddings.
        pos = torch.arange(0, T, device=device).unsqueeze(0)  # (1, T)
        # Cap positions to avoid index-out-of-range when generating beyond max_len.
        pos = pos.clamp(0, self.pos_embed.weight.size(0) - 1)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)

        # Pass through decoder blocks with optional KV cache.
        present_kv = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if (past_kv is not None and i < len(past_kv)) else None
            x, layer_kv = block(x, attention_mask, use_cache, layer_past)
            if use_cache:
                present_kv.append(layer_kv)

        x = self.norm(x)

        # LM head: predict next-token logits.
        logits = self.lm_head(x)

        return logits, present_kv

    def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                 top_k=40, eos_token=None):
        """
        Autoregressive text generation with KV caching.

        This is the core generation loop:
          1. Run the model on the prompt (first call, no cache).
          2. Take the logits of the LAST token.
          3. Apply temperature scaling and top-k filtering.
          4. Sample from the resulting distribution.
          5. Append the sampled token to the sequence.
          6. Repeat from step 2 with KV cache enabled.

        KV cache makes step 2 efficient: we only compute attention for the
        new token, reusing the cached K, V from all previous steps.

        Parameters:
          input_ids: (batch, seq_len) — the prompt
          max_new_tokens: how many tokens to generate
          temperature: higher = more random (1.0), lower = more deterministic (0.1)
          top_k: only sample from the k most likely tokens
          eos_token: stop generation when this token is generated

        Returns: (batch, prompt_len + generated_len) — full sequence
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # KV cache: starts empty. After the first forward pass on the prompt,
        # each layer stores its (K, V) for all prompt tokens.
        past_kv = None
        generated = input_ids

        for step in range(max_new_tokens):
            # On the first step, process the full prompt.
            # On subsequent steps, only process the last token (KV cache
            # handles the rest).
            if past_kv is None:
                model_input = generated
                use_cache = False
            else:
                model_input = generated[:, -1:]   # only the last token
                use_cache = True

            logits, past_kv = self(model_input, use_cache=use_cache, past_kv=past_kv)

            # Get logits for the LAST token (the one we just predicted).
            next_logits = logits[:, -1, :] / temperature

            # Top-k filtering: zero out all but the k most likely tokens.
            if top_k > 0:
                top_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = top_vals[:, -1].unsqueeze(-1)  # smallest among top-k
                next_logits[next_logits < threshold] = float("-inf")

            # Convert to probabilities and sample.
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to the generated sequence.
            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS token.
            if eos_token is not None and (next_token == eos_token).any():
                break

        return generated

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
