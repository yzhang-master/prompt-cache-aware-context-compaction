from __future__ import annotations


class ExactPrefixCache:
    """Caches complete token blocks keyed by their entire preceding prefix."""

    def __init__(self, block_size: int = 16) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._prefixes: set[tuple[str, ...]] = set()

    def lookup(self, tokens: tuple[str, ...]) -> int:
        matched = 0
        for end in range(self.block_size, len(tokens) + 1, self.block_size):
            if tuple(tokens[:end]) not in self._prefixes:
                break
            matched = end
        return matched

    def store(self, tokens: tuple[str, ...]) -> None:
        for end in range(self.block_size, len(tokens) + 1, self.block_size):
            self._prefixes.add(tuple(tokens[:end]))

    def __len__(self) -> int:
        return len(self._prefixes)
