"""
LSTM sentiment classification on IMDB.

Trains a character-level LSTM to classify movie reviews as positive or
negative. The model uses the last time step's hidden state for prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.lstm.model import LSTMSentiment


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for text, label in zip(texts, labels):
            tokens, mask = tokenizer.encode(text, max_len)
            self.examples.append({
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
                "labels": torch.tensor(label, dtype=torch.long),
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

    print("Loading IMDB...")
    ds_train = load_dataset("stanfordnlp/imdb", split="train")
    ds_test = load_dataset("stanfordnlp/imdb", split="test")

    # Shuffle to ensure both classes are present (IMDB is sorted: neg then pos).
    import random
    indices = list(range(len(ds_train["text"])))
    random.shuffle(indices)
    train_texts = [ds_train[i]["text"] for i in indices[:9000]]
    train_labels = [ds_train[i]["label"] for i in indices[:9000]]
    test_texts = ds_test["text"][:1000]
    test_labels = ds_test["label"][:1000]
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}")

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model = LSTMSentiment(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128, hidden_size=128, num_layers=1, num_classes=2,
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                _, predicted = torch.max(logits, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Loss: {train_loss/len(train_loader):.4f}  "
              f"Train Acc: {train_correct/train_total*100:.1f}%  "
              f"Test Acc: {test_correct/test_total*100:.1f}%")

    torch.save(model.state_dict(), "nlp/lstm/lstm_sentiment.pt")
    print(f"\nModel saved to nlp/lstm/lstm_sentiment.pt")


if __name__ == "__main__":
    train()
