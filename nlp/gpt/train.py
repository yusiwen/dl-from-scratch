"""
GPT autoregressive language model training on text8.

Training objective: predict the next token given all previous tokens.
Every position in the sequence is a training example — a single batch
of 32 sequences × 128 tokens generates 32 × 127 = 4064 predictions.

This is "self-supervised" learning: the data provides its own labels
(token[t] is the label for token[t-1]). No human annotation needed.

Dataset: text8 from HuggingFace (same as Word2Vec, already cached).
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.gpt.model import GPT


class TextDataset(Dataset):
    """
    Wraps text chunks into fixed-length sequences for autoregressive training.

    Each sample: tokens of length `max_len`.
    Labels: tokens shifted left by 1 (predict token[i+1] from token[i]).
    """

    def __init__(self, texts, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for text in texts:
            tokens, mask = tokenizer.encode(text, max_len)
            self.examples.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load text8.
    print("Loading text8 dataset...")
    ds = load_dataset("afmck/text8", split="train")
    raw = ds[0]["text"]
    chunk_size = 200
    chunks = [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
    chunks = chunks[:5000]
    print(f"  Total chunks: {len(chunks)}")

    dataset = TextDataset(chunks, tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=4, n_layers=4, max_len=128,
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)  # (batch, seq_len)

            # Autoregressive target: shift left by 1.
            # Input: tokens[0..n-1], Target: tokens[1..n]
            # This way, every position predicts the next token.
            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}  "
              f"PPL: {perplexity:.2f}")

    torch.save(model, "nlp/gpt/gpt_text8.pt")
    print(f"\nModel saved to nlp/gpt/gpt_text8.pt")


if __name__ == "__main__":
    train()
