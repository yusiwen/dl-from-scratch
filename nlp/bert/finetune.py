"""
BERT fine-tuning for sentiment classification (IMDB reviews).

After pre-training (MLM), the BERT encoder has learned contextual
representations. We add a small classification head on top of the [CLS]
token and fine-tune on labeled sentiment data.

Dataset: stanfordnlp/imdb from HuggingFace (25K/25K movie reviews).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.bert.model import BERTForClassification


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


def finetune():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = CharTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load IMDB from HuggingFace.
    print("Loading IMDB dataset...")
    ds_train = load_dataset("stanfordnlp/imdb", split="train")
    ds_test = load_dataset("stanfordnlp/imdb", split="test")

    # Use a subset for training speed.
    train_texts = ds_train["text"][:5000]
    train_labels = ds_train["label"][:5000]
    test_texts = ds_test["text"][:1000]
    test_labels = ds_test["label"][:1000]
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}")

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = BERTForClassification(
        vocab_size=tokenizer.vocab_size,
        num_classes=2, d_model=128, n_heads=4, n_layers=4, max_len=128,
    ).to(device)
    print(f"Model parameters: {model.num_params():,}")

    # Load pre-trained MLM weights if available.
    try:
        state_dict = torch.load("nlp/bert/bert_mlm.pt", map_location=device)
        bert_keys = {k.replace("bert.", ""): v for k, v in state_dict.items()
                     if k.startswith("bert.")}
        model.bert.load_state_dict(bert_keys, strict=False)
        print("Loaded pre-trained BERT weights")
    except FileNotFoundError:
        print("No pre-trained weights — training from scratch")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
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
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids, attention_mask)
                _, predicted = torch.max(logits, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Loss: {train_loss/len(train_loader):.4f}  "
              f"Train Acc: {train_correct/train_total*100:.1f}%  "
              f"Test Acc: {test_correct/test_total*100:.1f}%")

    torch.save(model.state_dict(), "nlp/bert/bert_finetuned.pt")
    print(f"\nModel saved to nlp/bert/bert_finetuned.pt")


if __name__ == "__main__":
    finetune()
