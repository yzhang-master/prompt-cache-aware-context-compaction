from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    (
        "stable-prefix-replaced-during-render",
        "src/prompt_cache_compaction/context.py",
        "return [*self._prefix, *middle, *self._history, current]",
        'return [Message("system", "mutated"), *middle, *self._history, current]',
        ["tests/test_context.py", "tests/test_bench.py"],
    ),
    (
        "compaction-drops-recent-complete-turns",
        "src/prompt_cache_compaction/context.py",
        "keep = min(len(self._history), self.config.keep_recent_turns * 2)",
        "keep = 0",
        ["tests/test_context.py"],
    ),
    (
        "semantic-numeric-evidence-boost-removed",
        "src/prompt_cache_compaction/semantic.py",
        "numeric_boost = (\n            3.0 if any(any(char.isdigit() for char in token) "
        "for token in words) else 0.0\n        )",
        "numeric_boost = 0.0",
        ["tests/test_semantic.py"],
    ),
    (
        "prefix-cache-lookup-always-misses",
        "src/prompt_cache_compaction/cache.py",
        "matched = end",
        "matched = 0",
        ["tests/test_cache.py", "tests/test_model.py"],
    ),
]


def main() -> None:
    records = []
    for name, relative_path, original, mutant, tests in MUTATIONS:
        path = ROOT / relative_path
        source = path.read_text(encoding="utf-8")
        if source.count(original) != 1:
            raise RuntimeError(f"mutation target is not unique: {name}")
        try:
            path.write_text(source.replace(original, mutant), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", *tests],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.write_text(source, encoding="utf-8")
        if completed.returncode == 0:
            raise RuntimeError(f"tests unexpectedly passed mutation: {name}")
        summary = completed.stdout.strip().splitlines()[-1]
        print(f"PASS (tests rejected): {name}: {summary}")
        records.append(
            {
                "mutation": name,
                "path": relative_path,
                "pytest_returncode": completed.returncode,
                "pytest_summary": summary,
            }
        )
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "restored_after_each_mutation": True,
        "mutations": records,
    }
    result_path = ROOT / "results" / "mutation.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
