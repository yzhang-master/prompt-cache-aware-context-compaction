from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from prompt_cache_compaction.tokens import Message, count_tokens, tokenize

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "were",
    "will",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class _Candidate:
    text: str
    index: int
    score: float
    terms: frozenset[str]


class SemanticCompactor:
    """Frequency/entity-scored extractive summaries with redundancy suppression."""

    def summarize(self, messages: list[Message], max_tokens: int) -> str:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        sentences = [
            sentence.strip()
            for message in messages
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", message.content)
            if sentence.strip()
        ]
        frequencies = Counter(
            token.lower()
            for sentence in sentences
            for token in tokenize(sentence)
            if token.lower() not in _STOPWORDS and token.isalnum()
        )
        candidates = [
            self._candidate(sentence, index, frequencies)
            for index, sentence in enumerate(sentences)
        ]
        selected: list[_Candidate] = []
        for candidate in sorted(candidates, key=lambda item: (-item.score, item.index)):
            if candidate.terms and any(
                self._jaccard(candidate.terms, other.terms) >= 0.9 for other in selected
            ):
                continue
            summary = " ".join(
                item.text for item in sorted([*selected, candidate], key=lambda x: x.index)
            )
            if count_tokens(summary) <= max_tokens:
                selected.append(candidate)
        return " ".join(item.text for item in sorted(selected, key=lambda x: x.index))

    @staticmethod
    def _candidate(text: str, index: int, frequencies: Counter[str]) -> _Candidate:
        words = [token for token in tokenize(text) if token.isalnum()]
        terms = frozenset(token.lower() for token in words if token.lower() not in _STOPWORDS)
        informative = [token.lower() for token in words if token.lower() not in _STOPWORDS]
        frequency_score = sum(math.log1p(frequencies[token]) for token in informative)
        frequency_score /= max(len(informative), 1)
        numeric_boost = (
            3.0 if any(any(char.isdigit() for char in token) for token in words) else 0.0
        )
        entity_boost = 1.0 if any(token[:1].isupper() for token in words) else 0.0
        return _Candidate(text, index, frequency_score + numeric_boost + entity_boost, terms)

    @staticmethod
    def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0
