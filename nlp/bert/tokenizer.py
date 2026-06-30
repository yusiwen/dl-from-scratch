"""
Character-level tokenizer with BERT special tokens.

Vocabulary:
  [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
  Then printable ASCII characters (space, !, ", ..., ~) = 5..100
  Plus newline (\n) = 101

Total vocab: ~100 tokens — small enough to train quickly on M4.
"""

import re


SPECIAL = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
}

ID_TO_SPECIAL = {v: k for k, v in SPECIAL.items()}


class CharTokenizer:
    """
    Maps each character to an integer ID and back.

    Tokenization is character-level: "cat" → [c, a, t] → [12, 5, 24].
    This is the simplest possible tokenization and demonstrates the core
    concept: converting raw text into numerical indices that the model
    can process via Embedding lookup.
    """

    def __init__(self):
        # Build character vocabulary from printable ASCII.
        self.char_to_id = dict(SPECIAL)
        self.id_to_char = dict(ID_TO_SPECIAL)

        # Add all printable ASCII characters (space through ~).
        for i in range(32, 127):
            ch = chr(i)
            idx = len(self.char_to_id)
            self.char_to_id[ch] = idx
            self.id_to_char[idx] = ch

        # Add newline.
        self.char_to_id["\n"] = len(self.char_to_id)
        self.id_to_char[len(self.id_to_char)] = "\n"

        self.vocab_size = len(self.char_to_id)
        self.pad_id = SPECIAL["[PAD]"]
        self.unk_id = SPECIAL["[UNK]"]
        self.cls_id = SPECIAL["[CLS]"]
        self.sep_id = SPECIAL["[SEP]"]
        self.mask_id = SPECIAL["[MASK]"]

    def encode(self, text, max_len=128):
        """
        Convert text to token IDs with [CLS] and [SEP].

        Output: [CLS, t1, t2, ..., tn, SEP, PAD, ...]  (length = max_len)
        """
        # Lowercase for consistency.
        text = text.lower().strip()
        tokens = [self.cls_id]
        for ch in text:
            tokens.append(self.char_to_id.get(ch, self.unk_id))
            if len(tokens) >= max_len - 1:
                break
        tokens.append(self.sep_id)

        # Pad to max_len.
        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        tokens = tokens + [self.pad_id] * (max_len - len(tokens))
        return tokens[:max_len], mask[:max_len]

    def decode(self, ids):
        """Convert token IDs back to readable text (for inspection only)."""
        chars = []
        for i in ids:
            if i in self.id_to_char:
                ch = self.id_to_char[i]
                if ch in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"):
                    ch = ch
                chars.append(ch)
            else:
                chars.append("[?]")
        return "".join(chars)
