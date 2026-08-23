"""Analysis of the obfuscation study outputs (STUDY_DESIGN.md section 9).

Produces, per phase:
* Sensitivity (recall) and false-positive rate by source x transform.
* Mean perplexity by source x transform (Phase 1 only).
* Paired perplexity test: original vs hash_replace (if scipy available).
* Logistic interaction bug_detected ~ source*transform + is_correct,
  clustered by problem_id (if statsmodels available).

scipy / statsmodels are optional; the descriptive tables always print.
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")


def load_phase(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        print(f"(skip) {path} not found")
        return None
    df = pd.DataFrame(json.loads(path.read_text()))
    # bug_detected may be None on parse failure; drop those for rate tables.
    return df


def rate_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity and FPR per source x transform.

    Positive-for-sensitivity = chunk has a bug (is_correct == False).
    Positive-for-FPR         = chunk is correct (is_correct == True) but flagged.
    """
    valid = df[df["bug_detected"].notna()].copy()
    valid["bug_detected"] = valid["bug_detected"].astype(bool)

    rows = []
    for (source, transform), g in valid.groupby(["source", "transform"]):
        buggy = g[~g["is_correct"]]
        correct = g[g["is_correct"]]
        sensitivity = buggy["bug_detected"].mean() if len(buggy) else float("nan")
        fpr = correct["bug_detected"].mean() if len(correct) else float("nan")
        rows.append(
            {
                "source": source,
                "transform": transform,
                "n": len(g),
                "n_buggy": len(buggy),
                "n_correct": len(correct),
                "sensitivity": sensitivity,
                "false_positive_rate": fpr,
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "transform"]).reset_index(drop=True)


def perplexity_table(df: pd.DataFrame) -> pd.DataFrame | None:
    if "perplexity_raw" not in df.columns:
        return None
    return (
        df.groupby(["source", "transform"])[["perplexity_raw", "perplexity_normalized"]]
        .mean()
        .reset_index()
    )


def paired_perplexity_test(df: pd.DataFrame) -> None:
    if "perplexity_raw" not in df.columns:
        return
    wide = df.pivot_table(
        index="chunk_id", columns="transform", values="perplexity_raw"
    )
    if not {"original", "hash_replace"}.issubset(wide.columns):
        return
    paired = wide[["original", "hash_replace"]].dropna()
    delta = paired["hash_replace"] - paired["original"]
    print("\nPaired perplexity (hash_replace - original):")
    print(f"  n={len(paired)}  mean delta={delta.mean():.3f}  median={delta.median():.3f}")
    try:
        from scipy import stats

        t, p = stats.wilcoxon(paired["hash_replace"], paired["original"])
        print(f"  Wilcoxon statistic={t:.1f}  p={p:.4g}")
    except ImportError:
        print("  (install scipy for the Wilcoxon test)")


def logistic_interaction(df: pd.DataFrame) -> None:
    valid = df[df["bug_detected"].notna()].copy()
    valid["bug_detected"] = valid["bug_detected"].astype(int)
    valid["is_correct"] = valid["is_correct"].astype(int)
    try:
        import statsmodels.formula.api as smf

        model = smf.logit(
            "bug_detected ~ C(source) * C(transform) + is_correct", data=valid
        ).fit(disp=False)
        cov = model.get_robustcov_results(cov_type="cluster", groups=valid["problem_id"])
        print("\nLogistic interaction (clustered by problem_id):")
        print(cov.summary())
    except ImportError:
        print("\n(install statsmodels for the logistic interaction model)")
    except Exception as e:  # e.g. perfect separation on a smoke-sized run
        print(f"\nLogistic model did not converge: {e}")


def report(name: str) -> None:
    df = load_phase(name)
    if df is None:
        return
    print(f"\n{'=' * 60}\n{name}  ({len(df)} records)\n{'=' * 60}")
    print("\nSensitivity / false-positive rate by source x transform:")
    print(rate_tables(df).to_string(index=False))

    ppl = perplexity_table(df)
    if ppl is not None:
        print("\nMean perplexity by source x transform:")
        print(ppl.to_string(index=False))

    paired_perplexity_test(df)
    logistic_interaction(df)


def main() -> None:
    report("obfuscation_phase1")
    report("obfuscation_phase2")


if __name__ == "__main__":
    main()
