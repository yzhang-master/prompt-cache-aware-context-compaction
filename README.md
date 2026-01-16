# prompt-cache-aware-context-compaction

> **Era:** 2025+ · Staff AI Engineer focus area (cache-aware agent context management). Technique from 2025+, reimplemented in 2026 using only the Python standard library.

## Problem

Long agent histories increase prompt work, but naive truncation loses task state and rewriting the whole
context defeats a prefix cache. This repository implements a context manager with a byte-stable system
prefix, budgeted semantic summaries of old turns, and complete recent turns. An exact token-prefix cache
and scripted model measure the resulting cache reuse, token cost, and answer retention on local traces.

## Method

```text
stable system messages | compacted semantic summary | last two complete turns | current user
                              ^                    ^
                              |                    +-- never split a user/assistant pair
                              +-- replace only when the 210-token budget is exceeded
```

The deterministic semantic compactor splits history into sentences, scores informative terms by
`log(1 + frequency)`, adds numeric-evidence and named-entity boosts, suppresses near-duplicates with
Jaccard similarity, and selects sentences under an 88-token summary budget. Selection is deterministic
and selected sentences return to chronological order. Existing summaries participate in later
compaction, keeping memory bounded instead of replaying all discarded turns.

Tokenization is an explicit regex over words and punctuation; each message adds role and end tokens.
The cache stores every complete 16-token prefix, keyed by **all preceding tokens**. A repeated block under
a different prefix is therefore not a false hit. Generated assistant tokens enter the cache, matching the
next request's serialized history.

The scripted task model answers project-deadline questions only by finding the fact in its actual prompt.
It has no access to the trace's expected answer. Synthetic accounting is:

```text
cost = (uncached input × 1.0 + cached input × 0.5 + output × 2.0) / 1,000,000 USD
```

These prices are an accounting scenario, not a provider quote.

## Results

Three local tasks, 36 model calls, 12 fact-retrieval questions, fresh cache per strategy. Raw results are
in `results/replay.json`.

| Context strategy | Cached prompt tokens | Mean prompt tokens | Correct answers | Successful tasks | Synthetic cost/task |
|---|---:|---:|---:|---:|---:|
| Full history | **88.28%** | 170.17 | 12/12 | 3/3 | $0.001301 |
| Sliding window, two turns | 66.18% | 102.75 | 0/12 | 0/3 | **$0.000945** |
| Stable-prefix semantic compaction | 79.53% | 152.00 | **12/12** | **3/3** | $0.001259 |

Stable compaction performed six compactions with zero stable-prefix violations. Compared with full
history, it reduced total prompt tokens by 10.68% and synthetic cost by 3.23%. It improved cache hit rate
by 13.35 percentage points over sliding-window history while retaining every tested answer.

The tradeoff is explicit: full history has the highest raw cache hit rate because it never rewrites old
messages. Compaction increased uncached prompt tokens by 55.99% on this small trace, but reduced total
billed tokens enough to lower cost under the pricing above. Sliding history is cheapest because it
discards the facts needed to complete the tasks; it is not an equally correct alternative.

All 11 tests pass locally. Mutation checks replaced the stable prefix, dropped recent complete turns,
removed numeric evidence scoring, and forced cache misses; each mutation made targeted tests fail.
Sources were restored after every mutation and the intact suite passed again. See `results/mutation.json`.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python scripts/replay.py --smoke   # one task, all strategies, no output file
python scripts/replay.py           # three tasks, rewrites results/replay.json
python scripts/mutation_check.py   # expects failures and restores every source file
```

Regenerate the checked-in local traces with `python scripts/generate_traces.py`.

## Limitations

- Semantic compaction is extractive lexical scoring, not embeddings, neural summarization, or an LLM.
  Important information that lacks salient terms, entities, or numbers can be lost.
- The scripted model tests only exact ISO project deadlines. Twelve questions across three synthetic
  tasks are a regression benchmark, not a general measure of task success or reasoning quality.
- The tokenizer and 16-token cache blocks are deterministic stand-ins, not a provider tokenizer or KV
  cache. No attention computation, latency saving, GPU memory saving, eviction policy, or concurrency is
  measured. Costs use synthetic prices.
- Cache hit percentage is token-weighted across requests. It does not by itself predict cost: rewriting
  summaries creates uncached tokens even while shortening prompts.
- Over-budget recent turns or an oversized stable prefix raise `ContextOverflowError`; the manager never
  silently splits a turn or changes the stable prefix.

## References

- I. Gim et al. *Prompt Cache: Modular Attention Reuse for Low-Latency Inference.* 2023.
  arXiv:2311.04934.
- H. Jiang et al. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models.*
  EMNLP 2023.
- L. Yang et al. *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt
  Compression.* 2023. arXiv:2310.06839.

## License

MIT
