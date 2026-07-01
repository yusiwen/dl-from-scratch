"""
GPT text generation with word-level tokenizer.

Demonstrates autoregressive generation with:
  - Word-level tokenization (each word = one token)
  - Temperature control: higher = more creative, lower = more focused
  - Top-k sampling: only consider the k most likely next tokens
  - KV cache: efficient generation (O(T) per step instead of O(T²))
  - [SEP] blocked: prevents special tokens from appearing in output
"""

import torch
from nlp.gpt.tokenizer import WordTokenizer
from nlp.gpt.model import GPT
from utils.device import get_device


def generate_text(model, tokenizer, prompt, max_new_tokens=50,
                  temperature=0.8, top_k=40):
    """Generate text from a prompt using word-level tokenizer."""
    tokens, _ = tokenizer.encode(prompt, max_len=64)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            bad_tokens_ids=[tokenizer.sep_id, tokenizer.pad_id, tokenizer.cls_id],
        )

    # Decode and clean.
    full_text = tokenizer.decode(output[0].tolist())
    return full_text.strip()


def evaluate():
    device = get_device()
    print(f"Device: {device}\n")

    model = torch.load("nlp/gpt/gpt_text8.pt", map_location="cpu", weights_only=False)
    model = model.to("cpu")
    print("Loaded model from nlp/gpt/gpt_text8.pt\n")

    tokenizer = WordTokenizer(vocab_size=5000)
    # We need the vocab to decode — build from a sample (loads mapping only).
    # The vocab is reconstructed from the model's embedding weights.
    # Since we can't rebuild the exact word list without the training data,
    # we rebuild from the saved model config.
    # For decode-only purposes, load the training sentences to rebuild vocab.
    from datasets import load_dataset
    ds = load_dataset("afmck/text8", split="train")
    raw = ds[0]["text"]
    sentences = raw.split(".")
    tokenizer.build_vocab(sentences)

    print("=" * 60)
    print("GPT Text Generation (word-level, temperature=0.8, top_k=40)")
    print("=" * 60)

    prompts = [
        "the meaning of life is",
        "the future of artificial intelligence is",
        "once upon a time there was a",
    ]

    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt,
                                  max_new_tokens=30, temperature=0.8, top_k=40)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")
        print()

    # Temperature comparison.
    print("\n" + "=" * 60)
    print("Temperature comparison (same prompt)")
    print("=" * 60)
    prompt = "the study of computer science"
    for temp in [0.3, 0.8, 1.2]:
        generated = generate_text(model, tokenizer, prompt,
                                  max_new_tokens=20, temperature=temp, top_k=40)
        print(f"  t={temp}: {generated}")


if __name__ == "__main__":
    evaluate()
