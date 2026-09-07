"""Stage 1: generate 240 recipes (48 conditions x 5 repeats).

Resumable: a (condition, repeat_index) already listed in the condition's
manifest.csv with an existing generation file is skipped, so a crash at
generation 150/240 resumes at 151 without re-spending on completed calls.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from api import MODEL, generate_recipe
from config import REPEATS, build_prompt, conditions
from storage import condition_dir, read_manifest, record_generation


def pending_jobs(repeats: int) -> list[tuple[str, str, str, int]]:
    jobs = []
    for recipe, adjective, scale in conditions():
        cdir = condition_dir(recipe, adjective, scale)
        manifest = read_manifest(cdir)
        for repeat_index in range(repeats):
            row = manifest.get(repeat_index)
            if row and (cdir / f"{row['hash']}.generation.json").exists():
                continue
            jobs.append((recipe, adjective, scale, repeat_index))
    return jobs


def run_one(job: tuple[str, str, str, int]) -> str:
    recipe, adjective, scale, repeat_index = job
    prompt = build_prompt(recipe, adjective, scale)
    result = generate_recipe(prompt)
    payload = {
        "recipe": recipe,
        "adjective": adjective,
        "scale": scale,
        "repeat_index": repeat_index,
        "prompt": prompt,
        "model": MODEL,
        "api_params": result["api_params"],
        "response_id": result["response_id"],
        "usage": result["usage"],
        "response_text": result["text"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return record_generation(condition_dir(recipe, adjective, scale), repeat_index, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N pending jobs (smoke tests).")
    args = parser.parse_args()

    jobs = pending_jobs(args.repeats)
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} generations pending ({args.repeats} repeats x 48 conditions)")

    done = 0
    # Manifest writes are guarded by a per-condition lock in storage.py.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:  # keep going; rerun picks up the gap
                print(f"  FAILED {job}: {exc}")
                continue
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"Done: {done}/{len(jobs)} generated")


if __name__ == "__main__":
    main()
