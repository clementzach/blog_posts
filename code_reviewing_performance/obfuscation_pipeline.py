"""Obfuscation / memorization / self-preference pipeline.

Builds a chunk registry of the 50 hardest MBPP problems x 4 sources, applies
the three obfuscation transforms, then runs:

* Phase 1 (local Qwen2.5-Coder-7B): per-chunk perplexity scoring only.
* Phase 2 (Gemini API): bug detection, self-preference focus.

Reuses the existing prompts/cache/harness: ``pipeline.review_code`` (which goes
through ``cached_call`` + ``parse_review_json``) for detection, ``harness``
for ground-truth correctness, and ``dataset.load_mbpp_sample`` for problems.
See STUDY_DESIGN.md.
"""

import argparse
import json
import tokenize
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from dataset import load_mbpp_sample
from harness import _last_defined_function, run_test
from pipeline import review_code
from select_hard_problems import select_hard_problems
from transforms import TRANSFORMS, apply_transform

RESULTS_DIR = Path("results")
MODEL_SOURCES = ("claude", "gpt", "gemini")  # from results.json
PHASE2_SOURCES = ("claude", "gemini", "human")
PHASE3_REVIEWERS = ("claude", "gpt", "sonnet")  # reviewers of human code, original only
REVIEW_PERSONA = "persona_1"


# ---------------------------------------------------------------------------
# Chunk registry
# ---------------------------------------------------------------------------

def _generated_code_index(results_path: Path) -> dict[tuple[str, int], tuple[str, bool]]:
    """(generator, problem_id) -> (pre_review_code, is_correct) from prior run."""
    results = json.loads(results_path.read_text())
    index: dict[tuple[str, int], tuple[str, bool]] = {}
    for r in results:
        key = (r["generator_model"], r["problem_id"])
        if key not in index:
            index[key] = (r["pre_review_code"], bool(r["successful_run_pre_review"]))
    return index


def build_chunk_registry(n_problems: int | None = None) -> list[dict]:
    """One chunk per (problem, source) for the 50 hardest problems.

    Model-source code and correctness are reused from ``results/results.json``;
    the ``human`` source is the MBPP canonical solution, labeled by running it
    through the test harness.
    """
    hard_ids = select_hard_problems()
    if n_problems is not None:
        hard_ids = hard_ids[:n_problems]
    hard_set = set(hard_ids)

    problems = {p["task_id"]: p for p in load_mbpp_sample(200) if p["task_id"] in hard_set}
    gen_index = _generated_code_index(RESULTS_DIR / "results.json")

    chunks: list[dict] = []
    for pid in hard_ids:
        problem = problems[pid]

        for source in MODEL_SOURCES:
            code, is_correct = gen_index[(source, pid)]
            chunks.append(_make_chunk(pid, source, code, is_correct, problem))

        human_code = problem["code"]
        expected_func = _last_defined_function(human_code)
        human_correct, _ = run_test(human_code, problem["test_list"], expected_func)
        chunks.append(_make_chunk(pid, "human", human_code, human_correct, problem))

    return chunks


def _make_chunk(pid: int, source: str, code: str, is_correct: bool, problem: dict) -> dict:
    return {
        "chunk_id": f"{pid}:{source}",
        "problem_id": pid,
        "source": source,
        "is_correct": is_correct,
        "code_original": code,
        "problem": problem,
    }


def transform_variants(chunk: dict) -> list[dict]:
    """Expand a chunk into its transform variants."""
    variants = []
    for transform in TRANSFORMS:
        try:
            code = apply_transform(chunk["code_original"], transform)
        except (SyntaxError, tokenize.TokenError):
            # Unparseable generation: fall back to the original text.
            code = chunk["code_original"]
        variants.append({**chunk, "transform": transform, "code": code})
    return variants


# ---------------------------------------------------------------------------
# Phase 1: local Qwen (perplexity only)
# ---------------------------------------------------------------------------

def run_phase1(chunks: list[dict]) -> list[dict]:
    from perplexity import score_perplexity

    variants = [v for c in chunks for v in transform_variants(c)]
    records: list[dict] = []
    total = len(variants)
    for i, v in enumerate(variants, 1):
        problem_text = v["problem"]["prompt"]
        ppl = score_perplexity(problem_text, v["code"])
        records.append(_detection_record(v, None, None, ppl))
        print(f"  [phase1 {i}/{total}] {v['chunk_id']} {v['transform']}", end="\r")
    print()
    return records


# ---------------------------------------------------------------------------
# Phase 2: Gemini API (detection)
# ---------------------------------------------------------------------------

def run_phase2(chunks: list[dict], max_workers: int = 50) -> list[dict]:
    variants = [
        v
        for c in chunks
        if c["source"] in PHASE2_SOURCES
        for v in transform_variants(c)
    ]
    records: list[dict] = []
    lock = Lock()
    total = len(variants)
    done = 0

    def _task(v: dict) -> dict:
        has_bug, parse_failure = review_code(
            "gemini", REVIEW_PERSONA, v["problem"], v["code"]
        )
        return _detection_record(v, has_bug, parse_failure, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, v) for v in variants]
        for future in as_completed(futures):
            records.append(future.result())
            with lock:
                done += 1
                print(f"  [phase2 {done}/{total}]", end="\r")
    print()
    return records


# ---------------------------------------------------------------------------
# Phase 3: Claude / GPT reviewing human code (original only)
# ---------------------------------------------------------------------------

def run_phase3(chunks: list[dict], max_workers: int = 50) -> list[dict]:
    """Each of PHASE3_REVIEWERS reviews every human chunk's original code.

    No obfuscation transforms: 50 human chunks x 2 reviewers = 100 detections.
    """
    human_chunks = [c for c in chunks if c["source"] == "human"]
    tasks = [(reviewer, c) for reviewer in PHASE3_REVIEWERS for c in human_chunks]
    records: list[dict] = []
    lock = Lock()
    total = len(tasks)
    done = 0

    def _task(reviewer: str, c: dict) -> dict:
        has_bug, parse_failure = review_code(
            reviewer, REVIEW_PERSONA, c["problem"], c["code_original"]
        )
        record = _detection_record(
            {**c, "transform": "original"}, has_bug, parse_failure, None
        )
        record["reviewer_model"] = reviewer
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_task, reviewer, c) for reviewer, c in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            with lock:
                done += 1
                print(f"  [phase3 {done}/{total}]", end="\r")
    print()
    return records


def _detection_record(v: dict, has_bug, parse_failure: bool, ppl: dict | None) -> dict:
    record = {
        "chunk_id": v["chunk_id"],
        "problem_id": v["problem_id"],
        "source": v["source"],
        "transform": v["transform"],
        "is_correct": v["is_correct"],
        "bug_detected": has_bug,
        "parse_failure": parse_failure,
    }
    if ppl is not None:
        record.update(
            perplexity_raw=ppl["perplexity_raw"],
            perplexity_normalized=ppl["perplexity_normalized"],
            mean_nll=ppl["mean_nll"],
            n_code_tokens=ppl["n_code_tokens"],
        )
    return record


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write(records: list[dict], name: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(records, indent=2))
    pd.DataFrame(records).to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    print(f"Wrote {len(records)} records to {RESULTS_DIR / name}.{{json,csv}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["1", "2", "3", "both", "all"], default="both")
    parser.add_argument("--n", type=int, default=None, help="Limit #problems (smoke test)")
    parser.add_argument("--max-workers", type=int, default=50)
    args = parser.parse_args()

    chunks = build_chunk_registry(n_problems=args.n)
    print(f"Built {len(chunks)} chunks ({len(chunks) // 4} problems x 4 sources)")

    if args.phase in ("1", "both", "all"):
        print("Phase 1: local Qwen perplexity scoring...")
        _write(run_phase1(chunks), "obfuscation_phase1")

    if args.phase in ("2", "both", "all"):
        print("Phase 2: Gemini detection...")
        _write(run_phase2(chunks, args.max_workers), "obfuscation_phase2")

    if args.phase in ("3", "all"):
        print("Phase 3: Claude / GPT reviewing human code (original only)...")
        _write(run_phase3(chunks, args.max_workers), "obfuscation_phase3")


if __name__ == "__main__":
    main()
