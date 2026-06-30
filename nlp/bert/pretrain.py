"""
BERT pre-training via Masked Language Model (MLM).

"Entropy increase noise reduction" in action:
  1. Take raw text → tokenize → randomly mask 15% of tokens (entropy increase).
  2. Transformer processes the corrupted sequence.
  3. MLM head predicts original tokens (denoising / noise reduction).
  4. Loss = cross-entropy on masked positions only.

Uses a small built-in sample dataset so no network download is required.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.bert.model import BERTForMLM


# Small sample texts (no external download needed).
SAMPLE_TEXTS = [
    "the cat sat on the mat and looked at the dog",
    "i love watching movies especially science fiction films",
    "the weather today is sunny and warm perfect for a walk",
    "machine learning is transforming how we process language",
    "the restaurant served delicious food with excellent service",
    "i cannot believe how terrible this product turned out to be",
    "the book was fascinating from the first page to the last",
    "my experience with customer support was frustrating and slow",
    "the concert last night was absolutely amazing and energetic",
    "this is the worst purchase i have ever made complete waste",
    "the team worked together to deliver an outstanding project",
    "i am extremely satisfied with the quality of this service",
    "the movie had great special effects but a confusing plot",
    "our vacation was wonderful the hotel was beautiful and clean",
    "the software keeps crashing and losing all my unsaved work",
] * 20  # 300 texts total


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
                # else: unchanged (10%)
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
    print(f"Sample texts: {len(SAMPLE_TEXTS)}")

    dataset = TextDataset(SAMPLE_TEXTS, tokenizer, max_len=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    model = BERTForMLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=4, n_layers=4, max_len=128,
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 20
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

    # Save for fine-tuning.
    torch.save(model.state_dict(), "nlp/bert/bert_mlm.pt")
    print(f"\nWeights saved to nlp/bert/bert_mlm.pt")


if __name__ == "__main__":
    pretrain()
