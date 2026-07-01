"""
LSTM sentiment classification on IMDB.

Trains a character-level LSTM to classify movie reviews as positive or
negative. The model uses the last time step's hidden state for prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.lstm.model import LSTMSentiment
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


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
    cfg = load_config("nlp/lstm/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print("Loading IMDB...")
    ds_train = load_dataset("stanfordnlp/imdb", split="train")
    ds_test = load_dataset("stanfordnlp/imdb", split="test")

    import random
    indices = list(range(len(ds_train["text"])))
    random.shuffle(indices)
    train_texts = [ds_train[i]["text"] for i in indices[:cfg["train_samples"]]]
    train_labels = [ds_train[i]["label"] for i in indices[:cfg["train_samples"]]]
    test_texts = ds_test["text"][:cfg["test_samples"]]
    test_labels = ds_test["label"][:cfg["test_samples"]]
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}")

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_len=cfg["max_len"])
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, max_len=cfg["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = LSTMSentiment(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg["embed_dim"], hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"], num_classes=cfg["num_classes"],
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/lstm")

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

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total * 100
        test_acc = test_correct / test_total * 100

        writer.add_scalar("train/loss", avg_train_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("test/acc", test_acc, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Loss: {avg_train_loss:.4f}  "
              f"Train Acc: {train_acc:.1f}%  "
              f"Test Acc: {test_acc:.1f}%")

    writer.close()
    save_path = "nlp/lstm/lstm_sentiment.pt"
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
