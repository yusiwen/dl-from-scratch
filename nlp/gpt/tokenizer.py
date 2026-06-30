"""
Word-level tokenizer for GPT.

Splits text on whitespace, builds vocabulary from the training corpus.
Each word becomes a single token, dramatically reducing sequence length
compared to character-level tokenization.

For example:
  Character-level: "the cat sat" → [t,h,e, ,c,a,t, ,s,a,t] (11 tokens)
  Word-level:      "the cat sat" → [the, cat, sat]          (3 tokens)

This makes it much easier for the model to learn semantic patterns.
Vocabulary is built from the text8 corpus, capped at 10000 most frequent words.
"""

from collections import Counter


SPECIAL = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
}

ID_TO_SPECIAL = {v: k for k, v in SPECIAL.items()}


class WordTokenizer:
    """
    Splits text into words, maps each word to an integer ID.

    encode(text) returns (token_ids, attention_mask) — same API as CharTokenizer
    so the rest of the code (train.py, generate.py) doesn't need to change.
    """

    def __init__(self, vocab_size=10000):
        self.vocab_size_limit = vocab_size
        self.word_to_id = dict(SPECIAL)
        self.id_to_word = dict(ID_TO_SPECIAL)
        self._fitted = False

    def build_vocab(self, texts):
        """Build vocabulary from a list of text strings (corpus)."""
        counter = Counter()
        for text in texts:
            for word in text.lower().split():
                counter[word] += 1

        # Keep the most frequent words.
        most_common = counter.most_common(self.vocab_size_limit - len(SPECIAL))
        for word, _ in most_common:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

        self.vocab_size = len(self.word_to_id)
        self._fitted = True
        return self

    @property
    def pad_id(self):
        return SPECIAL["[PAD]"]

    @property
    def unk_id(self):
        return SPECIAL["[UNK]"]

    @property
    def cls_id(self):
        return SPECIAL["[CLS]"]

    @property
    def sep_id(self):
        return SPECIAL["[SEP]"]

    @property
    def mask_id(self):
        return SPECIAL["[MASK]"]

    def encode(self, text, max_len=128):
        """
        Convert text to token IDs with [CLS] and [SEP].

        Output: [CLS, w1, w2, ..., wn, SEP, PAD, ...]  (length = max_len)
        """
        tokens = [self.cls_id]
        for word in text.lower().split():
            tokens.append(self.word_to_id.get(word, self.unk_id))
            if len(tokens) >= max_len - 1:
                break
        tokens.append(self.sep_id)

        mask = [1] * len(tokens) + [0] * (max_len - len(tokens))
        tokens = tokens + [self.pad_id] * (max_len - len(tokens))
        return tokens[:max_len], mask[:max_len]

    def decode(self, ids):
        """Convert token IDs back to readable text."""
        words = []
        for i in ids:
            if i in self.id_to_word:
                w = self.id_to_word[i]
                if w.startswith("[") and w.endswith("]"):
                    continue  # Skip special tokens in output.
                words.append(w)
            else:
                words.append("<?>")
        return " ".join(words)
