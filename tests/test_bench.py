from prompt_cache_compaction.bench import run_benchmark


def test_local_trace_smoke_compares_real_cache_and_cost_accounting() -> None:
    result = run_benchmark(smoke=True)
    full = result["strategies"]["full_history"]
    sliding = result["strategies"]["sliding_window"]
    stable = result["strategies"]["stable_compaction"]
    assert full["correct_answers"] == full["expected_answers"]
    assert stable["correct_answers"] == stable["expected_answers"]
    assert sliding["correct_answers"] < stable["correct_answers"]
    assert stable["compactions"] > 0
    assert stable["stable_prefix_violations"] == 0
    assert stable["total_cost_usd"] < full["total_cost_usd"]
    assert stable["prompt_tokens"] < full["prompt_tokens"]
    assert stable["prefix_cache_hit_rate_pct"] > sliding["prefix_cache_hit_rate_pct"]
    for metrics in result["strategies"].values():
        assert metrics["prompt_tokens"] == (
            metrics["cached_prompt_tokens"] + metrics["uncached_prompt_tokens"]
        )
        assert metrics["requests_with_cache_hit_pct"] > 0
