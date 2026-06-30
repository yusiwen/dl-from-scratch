"""
Word2Vec — CBOW and Skip-gram with Negative Sampling.

Word2Vec learns dense vector representations (embeddings) for words by
training on the task of predicting words from their context (CBOW) or
predicting context from words (Skip-gram).

Unlike BERT (which produces context-dependent representations), Word2Vec
produces a single static embedding per word. The embedding matrix is
essentially a lookup table: vocab_size × embed_dim.

The key innovation is Negative Sampling: instead of a full softmax over
the entire vocabulary (which is expensive), we train binary classifiers
that distinguish real (target, context) pairs from randomly sampled
(noise) pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(nn.Module):
    """
    Word2Vec with both CBOW and Skip-gram architectures.

    Two sets of embeddings:
      - target_embeddings:  used for the "center" word
      - context_embeddings: used for the "context" / "outside" words

    Having two separate embedding matrices is standard practice and
    improves training stability. The final word vectors are typically
    taken as the sum or average of both matrices.
    """

    def __init__(self, vocab_size, embed_dim=50):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Two separate embedding tables.
        self.target_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize with small values.
        nn.init.uniform_(self.target_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.uniform_(self.context_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def forward_cbow(self, context_ids, target_ids, noise_ids):
        """
        CBOW: average context embeddings → predict target word.

        context_ids: (batch, window_size)  — surrounding word indices
        target_ids:  (batch,)              — center word index (positive)
        noise_ids:   (batch, k)            — negative sample indices

        Returns the Negative Sampling loss (scalar).
        """
        batch_size = context_ids.size(0)

        # Average context embeddings → (batch, embed_dim).
        ctx_emb = self.context_embed(context_ids).mean(dim=1)

        # Positive score: how well does the average context predict the target?
        pos_target = self.target_embed(target_ids)         # (batch, embed_dim)
        pos_score = (ctx_emb * pos_target).sum(dim=1)      # (batch,)
        pos_loss = F.logsigmoid(pos_score).mean()

        # Negative scores: predict noise words instead.
        neg_target = self.target_embed(noise_ids)           # (batch, k, embed_dim)
        neg_score = (ctx_emb.unsqueeze(1) * neg_target).sum(dim=2)  # (batch, k)
        neg_loss = F.logsigmoid(-neg_score).mean()

        return -(pos_loss + neg_loss)

    def forward_skipgram(self, target_ids, context_ids, noise_ids):
        """
        Skip-gram: target embedding → predict context words.

        target_ids:  (batch,)              — center word index
        context_ids: (batch,)              — one context word index (positive)
        noise_ids:   (batch, k)            — negative sample indices

        Returns the Negative Sampling loss (scalar).
        """
        # Positive: target → context.
        tgt_emb = self.target_embed(target_ids)            # (batch, embed_dim)
        pos_ctx = self.context_embed(context_ids)          # (batch, embed_dim)
        pos_score = (tgt_emb * pos_ctx).sum(dim=1)
        pos_loss = F.logsigmoid(pos_score).mean()

        # Negative: target → noise words.
        neg_ctx = self.context_embed(noise_ids)            # (batch, k, embed_dim)
        neg_score = (tgt_emb.unsqueeze(1) * neg_ctx).sum(dim=2)
        neg_loss = F.logsigmoid(-neg_score).mean()

        return -(pos_loss + neg_loss)

    def get_embeddings(self):
        """
        Return the final word vectors (sum of target + context embeddings).
        """
        return (self.target_embed.weight + self.context_embed.weight).detach()
