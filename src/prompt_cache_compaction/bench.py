from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prompt_cache_compaction.cache import ExactPrefixCache
from prompt_cache_compaction.context import (
    ContextConfig,
    FullHistoryContext,
    SlidingWindowContext,
    StablePrefixContext,
)
from prompt_cache_compaction.model import CacheAwareScriptedModel, TokenPricing
from prompt_cache_compaction.tokens import Message, encode_messages

ROOT = Path(__file__).resolve().parents[2]
PRICING = TokenPricing(
    uncached_input_per_million=1.0,
    cached_input_per_million=0.5,
    output_per_million=2.0,
)


def make_context(strategy: str, prefix: list[Message]) -> Any:
    if strategy == "full_history":
        return FullHistoryContext(prefix)
    if strategy == "sliding_window":
        return SlidingWindowContext(prefix, keep_recent_turns=2)
    if strategy == "stable_compaction":
        return StablePrefixContext(
            prefix,
            ContextConfig(token_budget=210, summary_token_budget=88, keep_recent_turns=2),
        )
    raise ValueError(f"unknown strategy: {strategy}")


def replay_strategy(traces: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    model = CacheAwareScriptedModel(ExactPrefixCache(block_size=16), PRICING)
    requests = 0
    requests_with_hits = 0
    prompt_tokens = 0
    cached_tokens = 0
    uncached_tokens = 0
    completion_tokens = 0
    total_cost = 0.0
    compactions = 0
    prefix_violations = 0
    expected_answers = 0
    correct_answers = 0
    successful_tasks = 0
    task_results = []

    for trace in traces:
        prefix = [Message("system", trace["system"])]
        prefix_tokens = encode_messages(prefix)
        context = make_context(strategy, prefix)
        task_correct = True
        for turn in trace["turns"]:
            result = model.generate(context.prepare(turn["user"]))
            usage = result.usage
            requests += 1
            requests_with_hits += usage.cached_prompt_tokens > 0
            prompt_tokens += usage.prompt_tokens
            cached_tokens += usage.cached_prompt_tokens
            uncached_tokens += usage.uncached_prompt_tokens
            completion_tokens += usage.completion_tokens
            total_cost += usage.cost_usd
            prefix_violations += result.prompt_tokens[: len(prefix_tokens)] != prefix_tokens
            expected = turn["expected"]
            if expected is not None:
                expected_answers += 1
                correct = result.text == expected
                correct_answers += correct
                task_correct = task_correct and correct
            context.commit(turn["user"], result.text)
        compactions += context.compactions
        successful_tasks += task_correct
        task_results.append({"id": trace["id"], "all_expected_answers_correct": task_correct})

    return {
        "strategy": strategy,
        "tasks": len(traces),
        "requests": requests,
        "successful_tasks": successful_tasks,
        "task_success_rate_pct": 100 * successful_tasks / len(traces),
        "expected_answers": expected_answers,
        "correct_answers": correct_answers,
        "answer_accuracy_pct": 100 * correct_answers / expected_answers,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_tokens,
        "prefix_cache_hit_rate_pct": 100 * cached_tokens / prompt_tokens,
        "requests_with_cache_hit_pct": 100 * requests_with_hits / requests,
        "completion_tokens": completion_tokens,
        "mean_prompt_tokens": prompt_tokens / requests,
        "compactions": compactions,
        "stable_prefix_violations": prefix_violations,
        "total_cost_usd": total_cost,
        "cost_per_task_usd": total_cost / len(traces),
        "task_results": task_results,
    }


def run_benchmark(root: Path = ROOT, smoke: bool = False) -> dict[str, Any]:
    traces = json.loads((root / "data" / "traces.json").read_text(encoding="utf-8"))
    if smoke:
        traces = traces[:1]
    strategies = ["full_history", "sliding_window", "stable_compaction"]
    results = {name: replay_strategy(traces, name) for name in strategies}
    full = results["full_history"]
    stable = results["stable_compaction"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "deterministic rule-based project-deadline extractor; not a neural LLM",
        "cache": "exact chained token prefixes, 16-token complete blocks, generation appended",
        "pricing": {
            "units": "synthetic USD per million tokens, not a provider quote",
            "uncached_input": PRICING.uncached_input_per_million,
            "cached_input": PRICING.cached_input_per_million,
            "output": PRICING.output_per_million,
        },
        "context_config": {
            "token_budget": 210,
            "summary_token_budget": 88,
            "keep_recent_turns": 2,
        },
        "strategies": results,
        "stable_vs_full_cost_reduction_pct": 100
        * (1 - stable["total_cost_usd"] / full["total_cost_usd"]),
        "stable_vs_full_uncached_token_reduction_pct": 100
        * (1 - stable["uncached_prompt_tokens"] / full["uncached_prompt_tokens"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true", help="replay one task without writing results"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "replay.json")
    args = parser.parse_args()
    result = run_benchmark(smoke=args.smoke)
    if not args.smoke:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, metrics in result["strategies"].items():
        print(
            f"{name}: cache={metrics['prefix_cache_hit_rate_pct']:.2f}% "
            f"cost/task=${metrics['cost_per_task_usd']:.6f} "
            f"answers={metrics['correct_answers']}/{metrics['expected_answers']}"
        )


if __name__ == "__main__":
    main()
