from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Message:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {self.role}")


@lru_cache(maxsize=4096)
def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text))


def count_tokens(text: str) -> int:
    return len(tokenize(text))


def encode_message(message: Message) -> tuple[str, ...]:
    return (f"<{message.role}>", *tokenize(message.content), "<end>")


def encode_messages(messages: list[Message] | tuple[Message, ...]) -> tuple[str, ...]:
    return tuple(token for message in messages for token in encode_message(message))
