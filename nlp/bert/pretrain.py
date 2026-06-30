"""
BERT pre-training via Masked Language Model (MLM).

"Entropy increase noise reduction" in action:
  1. Take raw text → tokenize → randomly mask 15% of tokens (entropy increase).
  2. Transformer processes the corrupted sequence.
  3. MLM head predicts original tokens (denoising / noise reduction).
  4. Loss = cross-entropy on masked positions only.

Dataset: text8 from HuggingFace (~90M characters of cleaned Wikipedia).
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.bert.model import BERTForMLM


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.examples = []

        for text in texts:
            tokens, _ = tokenizer.encode(text, max_len)
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = list(self.examples[idx])
        labels = list(tokens)

        for i in range(len(tokens)):
            if tokens[i] in (self.tokenizer.cls_id, self.tokenizer.sep_id, self.tokenizer.pad_id):
                continue
            if torch.rand(1).item() < self.mask_prob:
                rand = torch.rand(1).item()
                if rand < 0.8:
                    tokens[i] = self.tokenizer.mask_id
                elif rand < 0.9:
                    tokens[i] = torch.randint(5, self.tokenizer.vocab_size, (1,)).item()
            else:
                labels[i] = -100

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if t != self.tokenizer.pad_id else 0 for t in tokens], dtype=torch.long
            ),
        }


def pretrain():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load text8 from HuggingFace.
    print("Loading text8 dataset...")
    ds = load_dataset("afmck/text8", split="train")
    raw = ds[0]["text"]  # One giant string (~90M chars).

    # Split into manageable chunks (each ≈ 1000 chars).
    chunk_size = 1000
    chunks = [raw[i:i + chunk_size] for i in range(0, len(raw), chunk_size)]
    # Use a subset for training speed.
    chunks = chunks[:5000]
    print(f"  Total chunks: {len(chunks)}")

    dataset = TextDataset(chunks, tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = BERTForMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=4, n_layers=4, max_len=128,
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}  "
              f"PPL: {perplexity:.2f}")

    torch.save(model.state_dict(), "nlp/bert/bert_mlm.pt")
    print(f"\nWeights saved to nlp/bert/bert_mlm.pt")


if __name__ == "__main__":
    pretrain()
