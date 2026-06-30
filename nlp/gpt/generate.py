"""
GPT text generation.

Demonstrates autoregressive generation with:
  - Temperature control: higher = more creative, lower = more focused
  - Top-k sampling: only consider the k most likely next tokens
  - KV cache: efficient generation (O(T) per step instead of O(T²))

The prompt is tokenized, fed to the model, and then the model generates
one token at a time, feeding each new token back as input.
"""

import torch
from nlp.bert.tokenizer import CharTokenizer
from nlp.gpt.model import GPT


def generate_text(model, tokenizer, prompt, max_new_tokens=200,
                  temperature=0.8, top_k=40):
    """Generate text from a prompt."""
    tokens, _ = tokenizer.encode(prompt, max_len=128)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode the full output and clean up special tokens.
    full_text = tokenizer.decode(output[0].tolist())
    # Remove special token markers.
    for token_name in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]:
        full_text = full_text.replace(token_name, "")
    # Extract only the generated part (after prompt, removing special tokens).
    words = full_text.split()
    generated = " ".join(words).strip()
    return generated


def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    tokenizer = CharTokenizer()

    model = torch.load("nlp/gpt/gpt_text8.pt", map_location="cpu", weights_only=False)
    model = model.to("cpu")
    # Run generation on CPU to avoid MPS placeholder storage bug.
    local_device = "cpu"

    print("=" * 60)
    print("GPT Text Generation (character-level, temperature=0.8, top_k=40)")
    print("=" * 60)

    prompts = [
        "the meaning of life is",
        "in the beginning",
        "the future of artificial intelligence",
        "once upon a time there was",
    ]

    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt,
                                  max_new_tokens=150, temperature=0.8, top_k=40)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:200]}...")
        print()

    # Temperature comparison.
    print("\n" + "=" * 60)
    print("Temperature comparison (same prompt)")
    print("=" * 60)
    prompt = "the study of language"
    for temp in [0.3, 0.8, 1.2]:
        generated = generate_text(model, tokenizer, prompt,
                                  max_new_tokens=100, temperature=temp, top_k=40)
        print(f"\nt={temp}: {generated[:100]}...")


if __name__ == "__main__":
    evaluate()
