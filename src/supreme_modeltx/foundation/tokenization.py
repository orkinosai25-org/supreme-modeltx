from __future__ import annotations


class CharacterTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, *, max_length: int) -> list[int]:
        encoded = [((ord(ch) % (self.vocab_size - 1)) + 1) for ch in text][:max_length]
        if not encoded:
            encoded = [1]
        if len(encoded) < max_length:
            encoded.extend([0] * (max_length - len(encoded)))
        return encoded

    def decode(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            if token_id <= 0:
                continue
            chars.append(chr(((token_id - 1) % 95) + 32))
        return "".join(chars)
