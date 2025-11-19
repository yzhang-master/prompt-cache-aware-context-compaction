from __future__ import annotations

from dataclasses import dataclass

from prompt_cache_compaction.semantic import SemanticCompactor
from prompt_cache_compaction.tokens import Message, encode_messages


class ContextOverflowError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextConfig:
    token_budget: int = 280
    summary_token_budget: int = 88
    keep_recent_turns: int = 2

    def __post_init__(self) -> None:
        if min(self.token_budget, self.summary_token_budget) < 1:
            raise ValueError("token budgets must be positive")
        if self.keep_recent_turns < 0:
            raise ValueError("keep_recent_turns cannot be negative")


class StablePrefixContext:
    def __init__(
        self,
        stable_prefix: list[Message],
        config: ContextConfig | None = None,
        compactor: SemanticCompactor | None = None,
    ) -> None:
        if not stable_prefix:
            raise ValueError("a stable system prefix is required")
        self._prefix = tuple(stable_prefix)
        self.config = config or ContextConfig()
        self._compactor = compactor or SemanticCompactor()
        self._history: list[Message] = []
        self._summary: Message | None = None
        self.compactions = 0

    @property
    def prefix_tokens(self) -> tuple[str, ...]:
        return encode_messages(list(self._prefix))

    @property
    def recent_history(self) -> tuple[Message, ...]:
        return tuple(self._history)

    @property
    def summary(self) -> str | None:
        return self._summary.content if self._summary else None

    def _render(self, current: Message) -> list[Message]:
        middle = [self._summary] if self._summary is not None else []
        return [*self._prefix, *middle, *self._history, current]

    def prepare(self, user_text: str) -> list[Message]:
        current = Message("user", user_text)
        prompt = self._render(current)
        if len(encode_messages(prompt)) > self.config.token_budget:
            self._compact()
            prompt = self._render(current)
        if len(encode_messages(prompt)) > self.config.token_budget:
            raise ContextOverflowError(
                "stable prefix, summary, recent turns, and input exceed budget"
            )
        return prompt

    def _compact(self) -> None:
        keep = min(len(self._history), self.config.keep_recent_turns * 2)
        old = self._history[: len(self._history) - keep]
        if not old:
            return
        source = ([self._summary] if self._summary is not None else []) + old
        summary = self._compactor.summarize(source, self.config.summary_token_budget)
        self._summary = Message("system", summary) if summary else None
        self._history = self._history[len(self._history) - keep :] if keep else []
        self.compactions += 1

    def commit(self, user_text: str, assistant_text: str) -> None:
        self._history.extend([Message("user", user_text), Message("assistant", assistant_text)])


class FullHistoryContext:
    def __init__(self, stable_prefix: list[Message]) -> None:
        self._prefix = tuple(stable_prefix)
        self._history: list[Message] = []
        self.compactions = 0

    def prepare(self, user_text: str) -> list[Message]:
        return [*self._prefix, *self._history, Message("user", user_text)]

    def commit(self, user_text: str, assistant_text: str) -> None:
        self._history.extend([Message("user", user_text), Message("assistant", assistant_text)])


class SlidingWindowContext:
    def __init__(self, stable_prefix: list[Message], keep_recent_turns: int = 2) -> None:
        self._prefix = tuple(stable_prefix)
        self._history: list[Message] = []
        self._keep_messages = keep_recent_turns * 2
        self.compactions = 0

    def prepare(self, user_text: str) -> list[Message]:
        if len(self._history) > self._keep_messages:
            self._history = self._history[-self._keep_messages :] if self._keep_messages else []
        return [*self._prefix, *self._history, Message("user", user_text)]

    def commit(self, user_text: str, assistant_text: str) -> None:
        self._history.extend([Message("user", user_text), Message("assistant", assistant_text)])
