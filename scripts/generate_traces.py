from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a project operations assistant. Answer only from facts retained in the conversation. "
    "Keep project names and ISO dates exact. When no answer is available say I do not know. "
    "The following conversation contains project updates and questions. Treat earlier facts as "
    "valid unless a later update explicitly changes them. Preserve deadlines during compaction."
)

TASKS = [
    (
        "launch",
        ["Apollo", "Borealis", "Cygnus", "Delphi", "Eos", "Flux"],
        ["2027-03-14", "2027-04-19", "2027-05-22", "2027-06-25", "2027-07-28", "2027-08-30"],
    ),
    (
        "maintenance",
        ["Gemini", "Helios", "Iris", "Janus", "Kronos", "Lyra"],
        ["2028-01-11", "2028-02-16", "2028-03-21", "2028-04-26", "2028-05-27", "2028-06-28"],
    ),
    (
        "migration",
        ["Mira", "Nova", "Orion", "Polaris", "Quasar", "Rhea"],
        ["2029-02-12", "2029-03-17", "2029-04-22", "2029-05-23", "2029-06-24", "2029-07-25"],
    ),
]


def make_turns(projects: list[str], dates: list[str]) -> list[dict[str, str | None]]:
    turns: list[dict[str, str | None]] = []
    for index, (project, date) in enumerate(zip(projects, dates, strict=True)):
        fact = f"Project {project} deadline is {date}."
        turns.append({"user": fact, "expected": None})
        if index == 1:
            turns.append(
                {
                    "user": "The review covered owners, milestones, and documentation.",
                    "expected": None,
                }
            )
        if index in {1, 2, 3}:
            question = f"What is the Project {projects[index - 1]} deadline?"
            previous_date = dates[index - 1]
            turns.append(
                {
                    "user": question,
                    "expected": f"Project {projects[index - 1]} deadline is {previous_date}.",
                }
            )
        if index == 3:
            turns.append(
                {
                    "user": "The team discussed release readiness and remaining checks.",
                    "expected": None,
                }
            )
    turns.append(
        {
            "user": f"What is the Project {projects[3]} deadline?",
            "expected": f"Project {projects[3]} deadline is {dates[3]}.",
        }
    )
    return turns


def main() -> None:
    traces = [
        {"id": task_id, "system": SYSTEM, "turns": make_turns(projects, dates)}
        for task_id, projects, dates in TASKS
    ]
    path = ROOT / "data" / "traces.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
