"""Select the "hard" MBPP problems from the already-cached prior-study results.

"Hard" = at least one of the two generator models (gpt, gemini) failed it,
computed from ``results/results.json`` so that no new generation is required.
See STUDY_DESIGN.md section 2.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
GENERATORS = ("gpt", "gemini")


def problem_pass_rates(results: list[dict]) -> dict[int, float]:
    """Mean pass rate per problem across the three generators.

    ``results.json`` contains one row per (generator, reviewer, persona,
    problem); ``successful_run_pre_review`` is constant for a given
    (generator, problem), so we deduplicate before averaging.
    """
    # (problem_id, generator) -> pass boolean, deduplicated.
    per_gen: dict[tuple[int, str], bool] = {}
    for r in results:
        key = (r["problem_id"], r["generator_model"])
        if key not in per_gen and r["successful_run_pre_review"] is not None:
            per_gen[key] = bool(r["successful_run_pre_review"])

    problem_ids = {pid for pid, _ in per_gen}
    rates: dict[int, float] = {}
    for pid in problem_ids:
        passes = [per_gen[(pid, g)] for g in GENERATORS if (pid, g) in per_gen]
        if passes:
            rates[pid] = sum(passes) / len(passes)
    return rates


def select_hard_problems(
    results_path: Path = RESULTS_DIR / "results.json", n: int | None = None
) -> list[int]:
    """Return problem ids where at least one generator got it wrong.

    That is, every ``pid`` with mean pass rate < 1.0 across ``GENERATORS``.
    Sorted by ascending pass rate (hardest first), ties broken by
    ``problem_id`` for determinism. If ``n`` is given, cap the result to the
    ``n`` hardest of those.
    """
    results = json.loads(Path(results_path).read_text())
    rates = problem_pass_rates(results)
    hard = sorted((pid for pid, rate in rates.items() if rate < 1.0), key=lambda pid: (rates[pid], pid))
    return hard[:n] if n is not None else hard


def main() -> None:
    results = json.loads((RESULTS_DIR / "results.json").read_text())
    rates = problem_pass_rates(results)
    hard = select_hard_problems()

    out = {
        "n_selected": len(hard),
        "task_ids": hard,
        "pass_rates": {str(pid): rates[pid] for pid in hard},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "hard_problems.json"
    out_path.write_text(json.dumps(out, indent=2))

    worst = hard[0]
    print(f"Selected {len(hard)} hardest problems -> {out_path}")
    print(f"  hardest: problem {worst} (mean pass rate {rates[worst]:.3f})")
    print(f"  easiest selected: problem {hard[-1]} (mean pass rate {rates[hard[-1]]:.3f})")


if __name__ == "__main__":
    main()
