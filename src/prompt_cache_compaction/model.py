from __future__ import annotations

import re
from dataclasses import dataclass

from prompt_cache_compaction.cache import ExactPrefixCache
from prompt_cache_compaction.tokens import Message, encode_message, encode_messages


@dataclass(frozen=True)
class TokenPricing:
    uncached_input_per_million: float = 1.0
    cached_input_per_million: float = 0.1
    output_per_million: float = 2.0


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    cached_prompt_tokens: int
    uncached_prompt_tokens: int
    completion_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class ModelResult:
    text: str
    usage: TokenUsage
    prompt_tokens: tuple[str, ...]


class CacheAwareScriptedModel:
    """Rule-based task model with exact token and prompt-cache accounting."""

    def __init__(
        self,
        cache: ExactPrefixCache | None = None,
        pricing: TokenPricing | None = None,
    ) -> None:
        self.cache = cache or ExactPrefixCache()
        self.pricing = pricing or TokenPricing()

    def generate(self, messages: list[Message]) -> ModelResult:
        prompt_tokens = encode_messages(messages)
        cached = self.cache.lookup(prompt_tokens)
        uncached = len(prompt_tokens) - cached
        answer = self._answer(messages)
        completion_tokens = encode_message(Message("assistant", answer))
        self.cache.store((*prompt_tokens, *completion_tokens))
        cost = (
            uncached * self.pricing.uncached_input_per_million
            + cached * self.pricing.cached_input_per_million
            + len(completion_tokens) * self.pricing.output_per_million
        ) / 1_000_000
        usage = TokenUsage(len(prompt_tokens), cached, uncached, len(completion_tokens), cost)
        return ModelResult(answer, usage, prompt_tokens)

    @staticmethod
    def _answer(messages: list[Message]) -> str:
        question = messages[-1].content
        match = re.fullmatch(
            r"What is the (?:Project )?(?P<project>[A-Za-z][\w-]*) deadline\?", question
        )
        if match is None:
            return "Noted."
        project = match.group("project")
        context = "\n".join(message.content for message in messages[:-1])
        facts = re.finditer(
            r"Project\s+(?P<project>[A-Za-z][\w-]*)\s+deadline\s+is\s+"
            r"(?P<date>\d{4}-\d{2}-\d{2})\s*\.",
            context,
        )
        for fact in facts:
            if fact.group("project").lower() == project.lower():
                return f"Project {fact.group('project')} deadline is {fact.group('date')}."
        return "I do not know."
