"""Unicode tokenizer with Vietnamese-aware words and UTF-8 byte fallback.

Round-trips arbitrary Unicode. Frequent words stay whole; unknown words split
into `<bXX>` byte tokens so nothing is dropped.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
KNOW = "<know>"
USER = "<user>"
ASST = "<asst>"
SPECIALS = [PAD, BOS, EOS, UNK, KNOW, USER, ASST]


def byte_token(b: int) -> str:
    return f"<b{b:02x}>"


@dataclass
class Tokenizer:
    token_to_id: dict[str, int]
    id_to_token: list[str]

    @staticmethod
    def build(texts: list[str], max_words: int = 2800) -> "Tokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for tok in _split(text):
                counts[tok] = counts.get(tok, 0) + 1
        words = sorted(counts, key=lambda w: (-counts[w], w))[:max_words]
        vocab = list(SPECIALS) + [byte_token(i) for i in range(256)] + words
        token_to_id = {t: i for i, t in enumerate(vocab)}
        return Tokenizer(token_to_id=token_to_id, id_to_token=vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def know_id(self) -> int:
        return self.token_to_id[KNOW]

    @property
    def user_id(self) -> int:
        return self.token_to_id[USER]

    @property
    def asst_id(self) -> int:
        return self.token_to_id[ASST]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for tok in _split(text):
            idx = self.token_to_id.get(tok)
            if idx is not None:
                ids.append(idx)
                continue
            for b in tok.encode("utf-8"):
                ids.append(self.token_to_id[byte_token(b)])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        words: list[str] = []
        raw = bytearray()

        def flush() -> None:
            if raw:
                words.append(raw.decode("utf-8", errors="replace"))
                raw.clear()

        specials = set(SPECIALS)
        for i in ids:
            if i < 0 or i >= len(self.id_to_token):
                continue
            tok = self.id_to_token[i]
            if tok.startswith("<b") and tok.endswith(">") and len(tok) == 5:
                try:
                    raw.append(int(tok[2:4], 16))
                    continue
                except ValueError:
                    pass
            flush()
            if skip_special and tok in specials:
                continue
            words.append(tok)
        flush()
        return _detok(words)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"id_to_token": self.id_to_token}, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "Tokenizer":
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)["id_to_token"]
        return Tokenizer(token_to_id={t: i for i, t in enumerate(vocab)}, id_to_token=vocab)


def _split(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    return TOKEN_RE.findall(text)


_NO_SPACE_BEFORE = set(".,!?;:%)]}>»”’…-")
_NO_SPACE_AFTER = set("([{<«“‘-/")


def _detok(tokens: list[str]) -> str:
    out = ""
    for tok in tokens:
        if not tok:
            continue
        if not out:
            out = tok
            continue
        if tok in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER or tok.startswith("'"):
            out += tok
        else:
            out += " " + tok
    return out
