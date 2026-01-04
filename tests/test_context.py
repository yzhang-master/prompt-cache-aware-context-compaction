import pytest

from prompt_cache_compaction.context import (
    ContextConfig,
    ContextOverflowError,
    FullHistoryContext,
    SlidingWindowContext,
    StablePrefixContext,
)
from prompt_cache_compaction.tokens import Message, encode_messages


def populated_context() -> StablePrefixContext:
    context = StablePrefixContext(
        [
            Message(
                "system", "You are a careful project assistant. Preserve every project deadline."
            )
        ],
        ContextConfig(token_budget=155, summary_token_budget=80, keep_recent_turns=2),
    )
    for index, project in enumerate(["Apollo", "Borealis", "Cygnus", "Delphi", "Eos", "Flux"]):
        context.prepare(f"Project {project} deadline is 2027-0{index + 1}-14.")
        context.commit(f"Project {project} deadline is 2027-0{index + 1}-14.", "Noted.")
    return context


def test_compaction_preserves_byte_identical_stable_prefix() -> None:
    context = StablePrefixContext(
        [Message("system", "Stable instructions. Do not change this prefix.")],
        ContextConfig(token_budget=120, summary_token_budget=55, keep_recent_turns=1),
    )
    expected = context.prefix_tokens
    prompts = []
    for index in range(10):
        user = f"Project Item{index} deadline is 2027-02-{index + 10:02d}."
        prompt = context.prepare(user)
        prompts.append(encode_messages(prompt))
        context.commit(user, "Noted.")
    assert context.compactions >= 2
    for tokens in prompts:
        assert tokens[: len(expected)] == expected
        assert len(tokens) <= 120
    assert len(context.recent_history) == 4
    assert [message.role for message in context.recent_history] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "Item0" in (context.summary or "")


def test_budget_overflow_is_not_hidden_by_dropping_recent_turn() -> None:
    context = StablePrefixContext(
        [Message("system", "short prefix")],
        ContextConfig(token_budget=20, summary_token_budget=10, keep_recent_turns=1),
    )
    context.prepare("first input")
    context.commit("first input", "Noted.")
    with pytest.raises(ContextOverflowError):
        context.prepare("this current input is itself much too long for the configured budget")


def test_full_and_sliding_strategies_have_expected_history_behavior() -> None:
    prefix = [Message("system", "prefix")]
    full = FullHistoryContext(prefix)
    sliding = SlidingWindowContext(prefix, keep_recent_turns=1)
    for index in range(3):
        text = f"turn {index}"
        full.prepare(text)
        full.commit(text, "Noted.")
        sliding.prepare(text)
        sliding.commit(text, "Noted.")
    assert len(full.prepare("next")) == 8
    sliding_prompt = sliding.prepare("next")
    assert "turn 0" not in [message.content for message in sliding_prompt]
    assert "turn 2" in [message.content for message in sliding_prompt]
