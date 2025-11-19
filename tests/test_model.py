import pytest

from prompt_cache_compaction.cache import ExactPrefixCache
from prompt_cache_compaction.context import ContextConfig, StablePrefixContext
from prompt_cache_compaction.model import CacheAwareScriptedModel
from prompt_cache_compaction.tokens import Message


def test_model_recovers_fact_from_compact_prompt_and_accounts_for_cache() -> None:
    context = StablePrefixContext(
        [Message("system", "You answer project questions using known facts.")],
        ContextConfig(token_budget=125, summary_token_budget=70, keep_recent_turns=1),
    )
    model = CacheAwareScriptedModel(ExactPrefixCache(block_size=4))
    for index in range(5):
        text = f"Project Item{index} deadline is 2027-03-{index + 10}."
        result = model.generate(context.prepare(text))
        context.commit(text, result.text)
    question = "What is the Project Item0 deadline?"
    result = model.generate(context.prepare(question))
    assert result.text == "Project Item0 deadline is 2027-03-10."
    assert result.usage.cached_prompt_tokens > 0
    usage = result.usage
    assert usage.prompt_tokens == usage.cached_prompt_tokens + usage.uncached_prompt_tokens
    expected_cost = (
        usage.uncached_prompt_tokens
        + usage.cached_prompt_tokens * 0.1
        + usage.completion_tokens * 2
    ) / 1_000_000
    assert usage.cost_usd == pytest.approx(expected_cost)


def test_model_does_not_use_an_oracle_for_missing_facts() -> None:
    model = CacheAwareScriptedModel()
    messages = [Message("user", "What is the Project Missing deadline?")]
    assert model.generate(messages).text == "I do not know."
