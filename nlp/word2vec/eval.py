"""
Word2Vec evaluation: find similar words and visualise embedding space.

After training, the embedding matrix encodes semantic relationships:
similar words have similar vectors (measured by cosine similarity).

Examples of learned relationships:
  - "dog" ≈ "cat" (animals)
  - "good" ≈ "great" (positive adjectives)
  - "run" ≈ "running" (morphological variants)
"""

import torch
import torch.nn.functional as F


def load_vectors(path="nlp/word2vec/skipgram.pt"):
    """Load trained embeddings and vocabulary."""
    data = torch.load(path, map_location="cpu")
    return data["embeddings"], data["id_to_word"]


def cosine_similarities(query_vec, all_vecs):
    """Compute cosine similarity between query and all vectors."""
    query_vec = query_vec / query_vec.norm()
    all_vecs = all_vecs / all_vecs.norm(dim=1, keepdim=True)
    return all_vecs @ query_vec


def find_similar(word, embeddings, id_to_word, word_to_id, top_k=10):
    """Find the top_k most similar words to the query word."""
    if word not in word_to_id:
        print(f"  '{word}' not in vocabulary")
        return

    idx = word_to_id[word]
    query_vec = embeddings[idx]
    sims = cosine_similarities(query_vec, embeddings)

    # Get top-k (excluding the word itself).
    values, indices = sims.topk(top_k + 1)
    print(f"\n  Similar to '{word}':")
    for i in range(1, top_k + 1):
        similar_word = id_to_word[indices[i].item()]
        print(f"    {similar_word:<12} {values[i].item():.4f}")


def evaluate():
    embeddings, id_to_word = load_vectors()

    # Build reverse mapping.
    word_to_id = {w: i for i, w in enumerate(id_to_word)}

    vocab_size, embed_dim = embeddings.shape
    print(f"Embeddings: {vocab_size} words × {embed_dim} dims\n")

    # Test words (choose common English words from PTB).
    test_words = [
        "company", "market", "year",
        "good", "new", "last",
        "government", "bank", "president",
        "increase", "decrease", "change",
        "large", "small", "important",
        "work", "study", "help",
    ]

    for word in test_words:
        find_similar(word, embeddings, id_to_word, word_to_id)


if __name__ == "__main__":
    evaluate()
