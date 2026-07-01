"""
LSTM evaluation: predict sentiment and show example predictions.
"""

import torch
from datasets import load_dataset

from nlp.bert.tokenizer import CharTokenizer
from nlp.lstm.model import LSTMSentiment
from utils.device import get_device


def predict(text, model, tokenizer, device, max_len=128):
    """Predict sentiment (0=negative, 1=positive)."""
    tokens, mask = tokenizer.encode(text, max_len)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return pred, probs[pred].item()


def evaluate():
    device = get_device()
    print(f"Device: {device}")

    tokenizer = CharTokenizer()

    model = LSTMSentiment(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128, hidden_size=64, num_layers=1, num_classes=2,
    ).to(device)

    state_dict = torch.load("nlp/lstm/lstm_sentiment.pt", map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from nlp/lstm/lstm_sentiment.pt")
    print(f"Model parameters: {model.num_params():,}\n")

    test_texts = [
        "this movie was absolutely fantastic and i loved every minute of it",
        "a complete waste of time terrible acting and boring plot",
        "the cinematography was stunning but the story fell flat",
        "i cannot recommend this film enough it was a masterpiece",
        "one of the worst movies i have ever seen in my entire life",
    ]

    print("Sentiment Predictions")
    print("=" * 50)
    for text in test_texts:
        pred, prob = predict(text, model, tokenizer, device)
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"\n  [{sentiment}] ({prob:.1%} confidence)")
        print(f"  \"{text}\"")

    # Evaluate on a few IMDB test samples.
    print("\n\nIMDB Test Samples")
    print("=" * 50)
    ds_test = load_dataset("stanfordnlp/imdb", split="test")
    correct = 0
    n = 200
    for i in range(n):
        text = ds_test[i]["text"][:200]  # Use first 200 chars.
        label = ds_test[i]["label"]
        pred, _ = predict(text, model, tokenizer, device)
        correct += (pred == label)
    print(f"  Test acc on {n} samples: {correct/n:.1%}")


if __name__ == "__main__":
    evaluate()
