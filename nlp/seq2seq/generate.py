"""Translate English sentences to German using a trained Seq2Seq Transformer."""

import torch

from nlp.seq2seq.model import Transformer
from nlp.seq2seq.data import WordTokenizer, SPECIAL
from utils.config import load_config


def translate(model, tokenizer, text, max_len=64, device="cpu"):
    tokens = tokenizer.encode(text, max_len)
    src = torch.tensor([tokens], dtype=torch.long, device=device)
    src_mask = torch.ones_like(src)

    with torch.no_grad():
        generated = model.generate(src, src_mask, bos_idx=SPECIAL["[BOS]"], eos_idx=SPECIAL["[EOS]"])

    return tokenizer.decode(generated[0].tolist())


def demo():
    cfg = load_config("nlp/seq2seq/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Rebuild tokenizer from saved vocab (stubs — real vocab should be saved).
    # For proper inference, save tokenizer alongside model weights.
    tokenizer = WordTokenizer(vocab_size=cfg["vocab_size"])
    # Load sample data to rebuild vocab (simplified for demo).
    from nlp.seq2seq.data import load_multi30k
    _, test_loader, tokenizer = load_multi30k(
        batch_size=cfg["batch_size"], vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"], num_workers=0,
    )

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
        pad_idx=SPECIAL["[PAD]"],
    )
    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {cfg['model_path']}")

    examples = [
        "a man is playing guitar",
        "two dogs playing in the snow",
        "a girl is sitting on a bench",
    ]

    for text in examples:
        translation = translate(model, tokenizer, text, device=device)
        print(f"EN: {text}")
        print(f"DE: {translation}")
        print()


if __name__ == "__main__":
    demo()
