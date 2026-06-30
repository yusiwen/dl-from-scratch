"""
Word2Vec training on Penn Treebank dataset.

Builds a vocabulary from the PTB corpus, generates context-target pairs
with negative sampling, and trains CBOW and Skip-gram embedding models.
"""

import math
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from datasets import load_dataset

from nlp.word2vec.model import Word2Vec


def load_texts():
    """Load text8 dataset from HuggingFace (classic Word2Vec benchmark)."""
    ds = load_dataset("afmck/text8", split="train")
    # text8 is one giant string — split into sentences by period.
    text = ds[0]["text"]
    # Split into pseudo-sentences for context window generation.
    chunk_size = 1000
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def build_vocab(texts, min_count=3):
    """
    Build vocabulary from list of sentences.

    Returns:
      word_to_id: dict mapping word → integer index
      id_to_word: list mapping index → word
      counts: list of (word, count) sorted by frequency
    """
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    # Filter rare words.
    vocab = [(w, c) for w, c in counter.items() if c >= min_count]
    vocab.sort(key=lambda x: -x[1])

    word_to_id = {"[UNK]": 0}
    id_to_word = ["[UNK]"]
    for word, _ in vocab:
        word_to_id[word] = len(id_to_word)
        id_to_word.append(word)

    return word_to_id, id_to_word, vocab


def subsample(texts, word_to_id, t=1e-4):
    """
    Subsample frequent words (Mikolov et al. 2013).

    Very frequent words (like "the", "a", "of") carry little semantic
    meaning but appear in many context windows. We probabilistically
    discard them during training with probability:

        P(discard) = 1 - sqrt(t / f(word))

    where f(word) is the relative frequency. This speeds up training
    and produces better embeddings.
    """
    total = sum(sum(1 for _ in text.split()) for text in texts)
    freqs = {}
    for text in texts:
        for w in text.lower().split():
            if w in word_to_id:
                freqs[w] = freqs.get(w, 0) + 1

    subsampled = []
    for text in texts:
        words = text.lower().split()
        filtered = []
        for w in words:
            if w not in word_to_id:
                continue
            f = freqs[w] / total
            p = 1.0 - math.sqrt(t / f) if f > t else 0.0
            if random.random() > p:
                filtered.append(word_to_id[w])
        if len(filtered) > 1:
            subsampled.append(filtered)
    return subsampled


def generate_training_pairs(token_ids, window_size=2):
    """
    Generate (target, context) pairs for Skip-gram.

    For each position i, consider a window of ±window_size around it.
    Each (word_i, word_j) for j in the window is a positive training pair.

    Also returns context words for CBOW (all context words around a target).
    """
    cbow_pairs = []  # (context_ids, target_id)
    skipgram_pairs = []  # (target_id, context_id)

    for sentence in token_ids:
        length = len(sentence)
        for i, target in enumerate(sentence):
            start = max(0, i - window_size)
            end = min(length, i + window_size + 1)
            context = []
            for j in range(start, end):
                if j == i:
                    continue
                context.append(sentence[j])
                skipgram_pairs.append((target, sentence[j]))
            if context:
                cbow_pairs.append((context, target))

    return cbow_pairs, skipgram_pairs


class NoiseSampler:
    """
    Samples negative examples from a unigram distribution raised to 3/4 power.

    P(w) = count(w)^0.75 / Σ count(w)^0.75

    This smoothed distribution empirically produces better embeddings
    than the raw frequency distribution.
    """

    def __init__(self, counts, vocab_size):
        # counts is list of (word, count) for vocab_size-1 words (excluding [UNK]).
        weights = np.array([c ** 0.75 for _, c in counts]) + 1e-8
        # Add a small weight for [UNK].
        weights = np.concatenate([[1e-8], weights])
        weights /= weights.sum()
        self.sampler = np.random.choice(vocab_size, size=100000, p=weights)
        self.pos = 0

    def sample(self, batch_size, k):
        """Return (batch_size, k) noise indices."""
        if self.pos + batch_size * k >= len(self.sampler):
            self.pos = 0
        noise = self.sampler[self.pos:self.pos + batch_size * k].reshape(batch_size, k)
        self.pos += batch_size * k
        return torch.tensor(noise, dtype=torch.long)


class CBOWDataset(Dataset):
    def __init__(self, pairs, max_window=2):
        self.pairs = pairs
        self.max_ctx = max_window * 2

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ctx, tgt = self.pairs[idx]
        # Pad context to fixed size (pad with 0 = [UNK]).
        if len(ctx) < self.max_ctx:
            ctx = ctx + [0] * (self.max_ctx - len(ctx))
        return torch.tensor(ctx[:self.max_ctx], dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


class SkipGramDataset(Dataset):
    def __init__(self, pairs, vocab_size):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tgt, ctx = self.pairs[idx]
        return torch.tensor(tgt, dtype=torch.long), torch.tensor(ctx, dtype=torch.long)


def train_epoch(model, loader, noise_sampler, optimizer, k=5, mode="skipgram"):
    """Train one epoch of Word2Vec."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        if mode == "skipgram":
            targets, contexts = batch
            # targets: (batch,), contexts: (batch,)
            noise = noise_sampler.sample(targets.size(0), k)
            loss = model.forward_skipgram(targets, contexts, noise)
        else:
            contexts, targets = batch
            # contexts: (batch, window), targets: (batch,)
            noise = noise_sampler.sample(targets.size(0), k)
            loss = model.forward_cbow(contexts, targets, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and build vocabulary.
    print("Loading PTB dataset...")
    texts = load_texts()
    print(f"  {len(texts)} sentences")

    word_to_id, id_to_word, vocab = build_vocab(texts, min_count=3)
    vocab_size = len(word_to_id)
    print(f"  Vocabulary size: {vocab_size}")

    # Subsample frequent words.
    print("Subsampling frequent words...")
    tokenized = subsample(texts, word_to_id)
    print(f"  After subsampling: {sum(len(s) for s in tokenized)} tokens")

    # Generate training pairs.
    print("Generating training pairs...")
    cbow_pairs, skipgram_pairs = generate_training_pairs(tokenized, window_size=2)
    print(f"  CBOW pairs: {len(cbow_pairs):,}")
    print(f"  Skip-gram pairs: {len(skipgram_pairs):,}")

    # Use a subset for speed.
    max_pairs = 200000
    cbow_pairs = cbow_pairs[:max_pairs]
    skipgram_pairs = skipgram_pairs[:max_pairs]

    # Build datasets and noise sampler.
    cbow_dataset = CBOWDataset(cbow_pairs, max_window=2)
    sg_dataset = SkipGramDataset(skipgram_pairs, vocab_size)
    cbow_loader = DataLoader(cbow_dataset, batch_size=128, shuffle=True, num_workers=0)
    sg_loader = DataLoader(sg_dataset, batch_size=128, shuffle=True, num_workers=0)

    noise_sampler = NoiseSampler(vocab, vocab_size)

    # --- Skip-gram ---
    print("\n" + "=" * 50)
    print("Skip-gram with Negative Sampling")
    print("=" * 50)

    model_sg = Word2Vec(vocab_size, embed_dim=50)
    optimizer = optim.Adam(model_sg.parameters(), lr=0.01)

    for epoch in range(1, 6):
        loss = train_epoch(model_sg, sg_loader, noise_sampler, optimizer, k=5, mode="skipgram")
        print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # Save Skip-gram embeddings.
    emb_sg = model_sg.get_embeddings()
    torch.save({
        "embeddings": emb_sg,
        "id_to_word": id_to_word,
        "vocab_size": vocab_size,
        "embed_dim": 50,
    }, "nlp/word2vec/skipgram.pt")
    print(f"\n  Skip-gram embeddings saved to nlp/word2vec/skipgram.pt")

    # --- CBOW ---
    print("\n" + "=" * 50)
    print("CBOW with Negative Sampling")
    print("=" * 50)

    model_cbow = Word2Vec(vocab_size, embed_dim=50)
    optimizer = optim.Adam(model_cbow.parameters(), lr=0.01)

    for epoch in range(1, 6):
        loss = train_epoch(model_cbow, cbow_loader, noise_sampler, optimizer, k=5, mode="cbow")
        print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # Save CBOW embeddings.
    emb_cbow = model_cbow.get_embeddings()
    torch.save({
        "embeddings": emb_cbow,
        "id_to_word": id_to_word,
        "vocab_size": vocab_size,
        "embed_dim": 50,
    }, "nlp/word2vec/cbow.pt")
    print(f"\n  CBOW embeddings saved to nlp/word2vec/cbow.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    train()
