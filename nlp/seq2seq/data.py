"""Multi30k dataset for EN→DE translation with word-level tokenizer."""

from collections import Counter
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader


SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL.items()}


class WordTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size_limit = vocab_size
        self.word_to_id = dict(SPECIAL)
        self.id_to_word = dict(ID_TO_SPECIAL)

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            for word in text.lower().split():
                counter[word] += 1
        most_common = counter.most_common(self.vocab_size_limit - len(SPECIAL))
        for word, _ in most_common:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
        self.vocab_size = len(self.word_to_id)

    def encode(self, text, max_len=64):
        tokens = [SPECIAL["[BOS]"]]
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, SPECIAL["[UNK]"]))
            if len(tokens) >= max_len - 1:
                break
        tokens.append(SPECIAL["[EOS]"])
        return tokens[:max_len]

    def decode(self, ids):
        words = []
        for i in ids:
            if i in self.id_to_word:
                w = self.id_to_word[i]
                if w.startswith("[") and w.endswith("]"):
                    continue
                words.append(w)
        return " ".join(words)


def collate_fn(batch, pad_idx=0):
    src, tgt = zip(*batch)
    src_len = max(len(s) for s in src)
    tgt_len = max(len(t) for t in tgt)

    src_padded = torch.full((len(batch), src_len), pad_idx, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_len), pad_idx, dtype=torch.long)
    src_mask = torch.zeros((len(batch), src_len), dtype=torch.long)

    for i, (s, t) in enumerate(zip(src, tgt)):
        src_padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_padded[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        src_mask[i, :len(s)] = 1

    return src_padded, tgt_padded, src_mask


def load_multi30k(batch_size=64, vocab_size=10000, max_len=64, num_workers=4):
    print("Loading Multi30k EN→DE...")
    ds = load_dataset("bentrevett/multi30k", split="train")
    test_ds = load_dataset("bentrevett/multi30k", split="test")

    # Filter out samples that exceed max_len after tokenization.
    en_texts = [item["en"] for item in ds]
    de_texts = [item["de"] for item in ds]
    test_en = [item["en"] for item in test_ds]
    test_de = [item["de"] for item in test_ds]

    tokenizer = WordTokenizer(vocab_size)
    tokenizer.build_vocab(de_texts)
    print(f"DE vocabulary: {tokenizer.vocab_size:,}")

    train_pairs = [(tokenizer.encode(en, max_len), tokenizer.encode(de, max_len))
                   for en, de in zip(en_texts, de_texts)]
    test_pairs = [(tokenizer.encode(en, max_len), tokenizer.encode(de, max_len))
                  for en, de in zip(test_en, test_de)]

    class _Dataset(Dataset):
        def __init__(self, pairs):
            self.pairs = pairs
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            s, t = self.pairs[idx]
            return torch.tensor(s, dtype=torch.long), torch.tensor(t, dtype=torch.long)

    train_dataset = _Dataset(train_pairs)
    test_dataset = _Dataset(test_pairs)

    pad_idx = SPECIAL["[PAD]"]
    from functools import partial

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=partial(collate_fn, pad_idx=pad_idx),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=partial(collate_fn, pad_idx=pad_idx),
    )

    return train_loader, test_loader, tokenizer
