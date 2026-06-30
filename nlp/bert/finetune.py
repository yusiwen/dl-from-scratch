"""
BERT fine-tuning for sentiment classification.

After pre-training (MLM), the BERT encoder has learned contextual
representations. We add a small classification head on top of the [CLS]
token and fine-tune on labeled sentiment data.

Uses built-in sample texts so no external download is required.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nlp.bert.tokenizer import CharTokenizer
from nlp.bert.model import BERTForClassification


# Labeled sentiment samples (expanded for meaningful training).
POSITIVE_TEXTS = [
    "this movie was absolutely fantastic and i loved every minute",
    "the service was excellent and the staff were very friendly",
    "i am extremely happy with my purchase it exceeded expectations",
    "the food was delicious and the atmosphere was wonderful",
    "a brilliant performance from start to finish highly recommended",
    "the product quality is outstanding and worth every penny",
    "our stay was perfect the room was clean and comfortable",
    "i love this app it is intuitive and works flawlessly",
    "the customer support team was helpful and resolved my issue quickly",
    "an amazing experience i will definitely come back again",
    "the design is beautiful and the build quality is top notch",
    "this is the best purchase i have made all year",
    "the staff went above and beyond to make our evening special",
    "i am very satisfied with the results exceeded all expectations",
    "a wonderful read from cover to cover i could not put it down",
    "the interface is user friendly and the features are exactly what i needed",
    "absolutely love this place the coffee is the best in town",
    "fast shipping and the item arrived in perfect condition",
    "the teacher was knowledgeable and made the class engaging",
    "i recommend this to everyone looking for quality and value",
]

NEGATIVE_TEXTS = [
    "this was a complete waste of time terrible experience overall",
    "the product arrived damaged and customer service was unhelpful",
    "i regret buying this it broke within a week of purchase",
    "the food was cold and the service was incredibly slow",
    "a poorly made film with bad acting and a confusing plot",
    "the software crashes constantly and loses all my work",
    "i had a terrible experience the staff was rude and unprofessional",
    "this is the worst quality i have ever seen do not buy it",
    "the hotel room was dirty and the bed was uncomfortable",
    "disappointing results after waiting so long for delivery",
    "the customer support team was unresponsive and did not help at all",
    "i cannot believe how overpriced this product is for its quality",
    "the movie was boring and predictable i almost fell asleep",
    "terrible customer service i will never use this company again",
    "the app keeps freezing and the latest update made it worse",
    "a frustrating experience from start to finish very disappointing",
    "the quality has gone downhill significantly since i last ordered",
    "i asked for a refund but they refused citing a vague policy",
    "the packaging was poor and the item arrived with scratches",
    "not worth the money at all there are much better alternatives",
]

# Double the dataset with simple variations.
SENTIMENT_TEXTS = (
    [(1, t) for t in POSITIVE_TEXTS] +
    [(0, t) for t in NEGATIVE_TEXTS] +
    [(1, t.replace("very", "really")) for t in POSITIVE_TEXTS[:10]] +
    [(1, t.replace("absolutely", "truly")) for t in POSITIVE_TEXTS[:10]] +
    [(0, t.replace("terrible", "awful")) for t in NEGATIVE_TEXTS[:10]] +
    [(0, t.replace("worst", "most disappointing")) for t in NEGATIVE_TEXTS[:10]]
)


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for label, text in data:
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

    import random
    random.seed(42)
    random.shuffle(SENTIMENT_TEXTS)

    # Stratified 80/20 split.
    pos = [(l, t) for l, t in SENTIMENT_TEXTS if l == 1]
    neg = [(l, t) for l, t in SENTIMENT_TEXTS if l == 0]
    split_pos = int(len(pos) * 0.8)
    split_neg = int(len(neg) * 0.8)
    train_data = pos[:split_pos] + neg[:split_neg]
    test_data = pos[split_pos:] + neg[split_neg:]
    random.shuffle(train_data)
    random.shuffle(test_data)
    print(f"Train: {len(train_data)}  Test: {len(test_data)}")

    train_dataset = SentimentDataset(train_data, tokenizer)
    test_dataset = SentimentDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

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

    num_epochs = 100
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
