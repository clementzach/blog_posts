import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pipeline import RESULTS_DIR, run_pipeline


def main() -> None:
    results = run_pipeline()

    json_path = RESULTS_DIR / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} records to {json_path}")

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV to {csv_path}")

    total = len(results)
    bugs = sum(1 for r in results if r["bug_detected"] is True)
    parse_fail_review = sum(1 for r in results if r["parse_failure_review"])
    parse_fail_gen = sum(1 for r in results if r["parse_failure_generation"])
    parse_fail_regen = sum(1 for r in results if r["parse_failure_regeneration"])

    print(f"\nSummary:")
    print(f"  Total records:                   {total}")
    print(f"  Bugs detected by reviewer:       {bugs}")
    print(f"  Parse failures (generation):     {parse_fail_gen}")
    print(f"  Parse failures (review JSON):    {parse_fail_review}")
    print(f"  Parse failures (fix code fence): {parse_fail_regen}")


if __name__ == "__main__":
    main()
