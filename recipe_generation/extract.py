"""Stage 2: one extraction call per cached generation (1:1, 240 total).

Resumable: skips any generation that already has a {hash}.extraction.json.

Changing EXTRACTION_SYSTEM_PROMPT does NOT invalidate the cache on its own --
delete the stale {hash}.extraction.json files and re-run. score.py will refuse
to score any extraction left behind at an older schema version, so a forgotten
delete shows up as a status, not as a wrong number.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from api import MODEL, extract_ingredients
from config import EXTRACTION_SCHEMA_VERSION
from storage import iter_generations, read_json, write_json


def pending_jobs():
    jobs = []
    for cdir, repeat_index, gen_hash in iter_generations():
        if (cdir / f"{gen_hash}.extraction.json").exists():
            continue
        jobs.append((cdir, repeat_index, gen_hash))
    return jobs


def run_one(job) -> None:
    cdir, repeat_index, gen_hash = job
    generation = read_json(cdir / f"{gen_hash}.generation.json")
    result = extract_ingredients(generation["response_text"])
    write_json(
        cdir / f"{gen_hash}.extraction.json",
        {
            "hash": gen_hash,
            "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
            "recipe": generation["recipe"],
            "adjective": generation["adjective"],
            "scale": generation["scale"],
            "repeat_index": repeat_index,
            "model": MODEL,
            "api_params": result["api_params"],
            "response_id": result["response_id"],
            "usage": result["usage"],
            "extractor_raw": result["raw"],
            "extraction": result["parsed"],
            "parse_failure": result["parsed"] is None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    jobs = pending_jobs()
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} extractions pending")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"  FAILED {job[0].name} r{job[1]}: {exc}")
                continue
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}")
    print(f"Done: {done}/{len(jobs)} extracted")


if __name__ == "__main__":
    main()
