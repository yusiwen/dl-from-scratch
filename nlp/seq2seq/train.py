import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from nlp.seq2seq.model import Transformer
from nlp.seq2seq.data import load_multi30k, SPECIAL
from utils.config import load_config, save_config
from utils.seed import set_seed
from utils.device import get_device


def train():
    cfg = load_config("nlp/seq2seq/config.yaml")
    set_seed(cfg["seed"])

    device = get_device()
    print(f"Device: {device}")
    torch.set_num_threads(4)

    train_loader, test_loader, tokenizer = load_multi30k(
        batch_size=cfg["batch_size"], vocab_size=cfg["vocab_size"],
        max_len=cfg["max_len"], num_workers=cfg["num_workers"],
    )
    pad_idx = SPECIAL["[PAD]"]

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        max_len=cfg["max_len"], dropout=cfg["dropout"],
        pad_idx=pad_idx,
    ).to(device)
    print(f"Parameters: {model.num_params():,}")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    num_epochs = cfg["num_epochs"]
    writer = SummaryWriter(log_dir="runs/seq2seq")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for src, tgt, src_mask in train_loader:
            src, tgt, src_mask = src.to(device), tgt.to(device), src_mask.to(device)

            # Teacher forcing: decoder input = tgt[:-1], predict = tgt[1:].
            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:].contiguous()

            optimizer.zero_grad()
            logits = model(src, decoder_input, src_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        writer.add_scalar("train/loss", avg_loss, epoch)
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {avg_loss:.4f}")

        # Greedy decoding example.
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            src_example, _, src_mask_example = next(iter(test_loader))
            src_example = src_example[:1].to(device)
            src_mask_example = src_mask_example[:1].to(device)

            with torch.no_grad():
                generated = model.generate(src_example, src_mask_example,
                                           bos_idx=SPECIAL["[BOS]"], eos_idx=SPECIAL["[EOS]"])

            src_text = " ".join(tokenizer.decode(src_example[0].tolist()))
            tgt_text = tokenizer.decode(generated[0].tolist())
            print(f"  EN: {src_text}")
            print(f"  DE: {tgt_text}")
            print()

    writer.close()
    save_path = cfg["model_path"]
    torch.save(model.state_dict(), save_path)
    save_config(cfg, save_path.replace(".pt", "_config.yaml"))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
