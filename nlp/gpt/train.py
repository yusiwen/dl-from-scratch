"""
GPT autoregressive language model training on text8.

Training objective: predict the next token given all previous tokens.
Every position in the sequence is a training example — a single batch
of 32 sequences × 128 tokens generates 32 × 127 = 4064 predictions.

Uses word-level tokenization: each word is one token (instead of each
character). This drastically reduces sequence length and helps the model
learn semantic patterns more effectively.

Dataset: text8 from HuggingFace (same as Word2Vec, already cached).
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from nlp.gpt.tokenizer import WordTokenizer
from nlp.gpt.model import GPT
from utils.config import load_config, save_config
from utils.seed import set_seed


class TextDataset(Dataset):
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
    cfg = load_config("nlp/gpt/config.yaml")
    set_seed(cfg["seed"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading text8 dataset...")
    ds = load_dataset("afmck/text8", split="train")
    raw = ds[0]["text"]

    words = raw.lower().split()
    chunk_words = cfg["chunk_words"]
    raw_chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    print(f"  Total chunks: {len(raw_chunks):,} ({len(words):,} words)")

    print("Building word-level vocabulary from text8...")
    tokenizer = WordTokenizer(vocab_size=cfg["vocab_size"])
    tokenizer.build_vocab(raw_chunks)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    sentences = [s.strip() for s in raw_chunks if len(s.strip()) > 5]
    sentences = sentences[:cfg["max_chunks"]]
    print(f"  Training chunks: {len(sentences):,}")

    dataset = TextDataset(sentences, tokenizer, max_len=cfg["max_len"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], max_len=cfg["max_len"],
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/gpt")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)

            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/perplexity", perplexity, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}  "
              f"PPL: {perplexity:.2f}")

    writer.close()
    save_path = "nlp/gpt/gpt_text8.pt"
    torch.save(model, save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
