"""LoRA fine-tuning of local GPT on a text8 subset."""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter

from nlp.gpt.tokenizer import WordTokenizer
from nlp.gpt.model import GPT
from nlp.lora.model import inject_lora, freeze_all_except_lora, lora_params_count
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("nlp/lora/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")

    # Load pretrained GPT.
    print(f"Loading pretrained GPT from {cfg['checkpoint']}...")
    model = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
    model = model.to(device)
    full_params = sum(p.numel() for p in model.parameters())
    print(f"Full model: {full_params:,} params")

    # Freeze all and inject LoRA.
    freeze_all_except_lora(model)
    model = inject_lora(model, r=cfg["r"], alpha=cfg["alpha"])
    lora_params = lora_params_count(model)
    print(f"LoRA params: {lora_params:,} ({lora_params / full_params:.2%} of full)")

    # Load text8 subset.
    print("Loading text8...")
    ds = load_dataset("afmck/text8", split="train")
    raw = ds[0]["text"]
    words = raw.lower().split()
    raw_chunks = [" ".join(words[i:i + 32]) for i in range(0, len(words), 32)]

    tokenizer = WordTokenizer(vocab_size=5000)
    tokenizer.build_vocab(raw_chunks)

    sentences = [s.strip() for s in raw_chunks if len(s.strip()) > 5]
    sentences = sentences[:cfg["max_chunks"]]
    print(f"Training chunks: {len(sentences):,}")

    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=64):
            self.examples = []
            for text in texts:
                tokens, mask = tokenizer.encode(text, max_len)
                self.examples.append({
                    "input_ids": torch.tensor(tokens, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                })
        def __len__(self):
            return len(self.examples)
        def __getitem__(self, idx):
            return self.examples[idx]

    dataset = TextDataset(sentences, tokenizer, max_len=cfg["max_len"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/lora")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            optimizer.zero_grad()
            logits, _ = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)

        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/perplexity", perplexity, epoch)

        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}  PPL: {perplexity:.2f}")

    writer.close()

    # Save only the LoRA parameters.
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}
    save_path = cfg["model_path"]
    torch.save(lora_state, save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"\nLoRA weights saved to {save_path} ({len(lora_state)} tensors)")


if __name__ == "__main__":
    train()
