"""
BERT evaluation: sentiment prediction + attention visualisation.

After fine-tuning, we can:
  1. Predict sentiment of new text.
  2. Visualise attention weights to see which tokens the model
     focuses on — the "semantic aggregation" in action.
"""

import torch
import torch.nn.functional as F

from nlp.bert.tokenizer import CharTokenizer
from nlp.bert.model import BERTForClassification
from utils.device import get_device


def predict(text, model, tokenizer, device):
    """Predict sentiment (0=negative, 1=positive) for a single text."""
    tokens, mask = tokenizer.encode(text, max_len=128)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    attention_mask = torch.tensor([mask], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits, attentions = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

    return pred, probs[pred].item(), attentions


def print_attention(text, attentions, tokenizer, layer=0, head=0):
    """
    Print attention weights for a specific layer and head.

    This shows which tokens the model attends to most strongly —
    the "semantic aggregation" pattern. Darker background = higher weight.
    """
    tokens, _ = tokenizer.encode(text, max_len=128)
    # Decode tokens to characters (skip [PAD]).
    chars = []
    for t in tokens:
        if t == tokenizer.pad_id:
            break
        ch = tokenizer.id_to_char.get(t, "?")
        chars.append(ch)

    # Get attention weights from the specified layer and head.
    attn = attentions[layer][0, head, :len(chars), :len(chars)]

    # Find the token with the highest attention to everything else.
    focus_idx = attn.sum(dim=1).argmax().item()

    print(f"\n  Layer {layer}, Head {head}: attention from '{chars[focus_idx]}'")
    print("  " + "-" * 70)
    weights = attn[focus_idx]
    for i, ch in enumerate(chars):
        bar_len = int(weights[i].item() * 50)
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {ch:>4} |{bar}| {weights[i]:.3f}")


def load_model(device):
    """Load the fine-tuned model."""
    tokenizer = CharTokenizer()
    model = BERTForClassification(
        vocab_size=tokenizer.vocab_size,
        num_classes=2,
        d_model=128,
        n_heads=4,
        n_layers=4,
        max_len=128,
    ).to(device)

    try:
        state_dict = torch.load("nlp/bert/bert_finetuned.pt", map_location=device)
        model.load_state_dict(state_dict)
        print("Loaded fine-tuned model from nlp/bert/bert_finetuned.pt")
    except FileNotFoundError:
        print("No fine-tuned model found. Run finetune.py first.")
        return None, None

    return model, tokenizer


def demo():
    device = get_device()
    print(f"Device: {device}\n")

    model, tokenizer = load_model(device)
    if model is None:
        return

    test_texts = [
        "this movie was absolutely fantastic and i loved every minute of it",
        "a complete waste of time terrible acting and boring plot",
        "the cinematography was stunning but the story fell flat",
    ]

    print("Sentiment Predictions")
    print("=" * 50)
    for text in test_texts:
        pred, prob, attentions = predict(text, model, tokenizer, device)
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"\n  [{sentiment}] ({prob:.1%} confidence)")
        print(f"  \"{text}\"")

    # Show attention for the first text.
    text = test_texts[0]
    _, _, attentions = predict(text, model, tokenizer, device)
    print(f"\n\nAttention Visualisation")
    print("=" * 50)
    print("(How the model aggregates semantic information)")
    print_attention(text, attentions, tokenizer, layer=3, head=1)


if __name__ == "__main__":
    demo()
