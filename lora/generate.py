"""Generate text using LoRA-adapted GPT."""

import torch

from nlp.gpt.model import GPT
from nlp.gpt.tokenizer import WordTokenizer
from lora.model import inject_lora, freeze_all_except_lora
from utils.config import load_config


def generate():
    cfg = load_config("lora/config.yaml")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load base model.
    print(f"Loading base GPT from {cfg['checkpoint']}...")
    model = torch.load(cfg["checkpoint"], map_location=device, weights_only=False)
    model = model.to(device)
    freeze_all_except_lora(model)

    # Inject LoRA and load trained weights.
    model = inject_lora(model, r=cfg["r"], alpha=cfg["alpha"])
    lora_state = torch.load(cfg["model_path"], map_location=device, weights_only=True)
    model.load_state_dict(lora_state, strict=False)
    model.eval()
    print(f"Loaded LoRA weights from {cfg['model_path']}")

    # Tokenizer.
    tokenizer = WordTokenizer(vocab_size=5000)
    from datasets import load_dataset
    ds = load_dataset("afmck/text8", split="train")
    raw = ds[0]["text"]
    sentences = raw.split(".")
    tokenizer.build_vocab(sentences)

    def generate_text(prompt, temperature=0.8, max_new=30):
        tokens, _ = tokenizer.encode(prompt, max_len=64)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=max_new,
                                    temperature=temperature, top_k=40,
                                    bad_tokens_ids=[tokenizer.sep_id, tokenizer.pad_id, tokenizer.cls_id])
        return tokenizer.decode(output[0].tolist()).strip()

    prompts = [
        "the meaning of life is",
        "artificial intelligence will change",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for temp in [0.3, 0.8, 1.2]:
            text = generate_text(prompt, temperature=temp)
            print(f"  t={temp:.1f}: {text}")


if __name__ == "__main__":
    generate()
