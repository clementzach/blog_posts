"""Stage 5: aggregate the scored generations.

Reports, per the study design:
  * noise floor  -- within-condition spread across the 5 repeats
  * adjective effect -- does "foolproof" tighten toward center, "avant garde"
    loosen, relative to the "really tasty" control?
  * scale effect -- does deviation grow from 6 -> 35 -> 173? Scale is treated
    as ordinal/continuous: `scale_trend` fits deviation against log10(people)
    across the three numeric conditions and reports a slope per decade with a
    bootstrap CI, rather than only comparing four buckets by eye. The
    unmodified prompt has no number, so it is off that axis by construction.
  * failure rate and deviation magnitude, REPORTED PER RECIPE. Pie dough and
    panna cotta use documented pass bands; pastry cream and choux use inferred
    +/-20% bands around a single tested ratio. Those are different kinds of
    ground truth, so failure rates are never pooled across all four.

Deviation is measured as |log2(value / band_center)|: symmetric for a ratio, so
"twice the center" and "half the center" score the same magnitude.

Deviation is the primary outcome and pass/fail is secondary, not the other way
round. On the pilot data three of the four recipes sat at a floor or a ceiling
-- choux and pastry cream failed almost every generation, panna cotta passed
every one -- so the binary measure had almost no variance for an adjective or a
serving size to explain, while the continuous measure still moved. Every rate
here is reported with a bootstrap CI for the same reason: at 5 repeats a cell
proportion of 0/5 or 5/5 is not evidence of much.
"""

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (ADJECTIVE_NAMES, GROUND_TRUTH, RECIPES, RESULTS_DIR,
                    SCALE_N, SCALE_NAMES)

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0

# Categorical slots 1-3 of the reference palette, in fixed order. Documented as
# clearing the all-pairs CVD and normal-vision floors, so they are safe for the
# point/line forms below. Never cycle or reorder these.
ADJ_COLOR = {"control": "#2a78d6", "foolproof": "#eb6834", "avant_garde": "#1baf7a"}
ADJ_MARKER = {"control": "o", "foolproof": "s", "avant_garde": "^"}
ADJ_LABEL = {"control": "really tasty (control)", "foolproof": "foolproof",
             "avant_garde": "avant garde"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
SCALE_LABEL = {"default": "no size\ngiven", "6": "6", "35": "35", "173": "173"}


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "scores.csv")
    df["abs_log2_deviation"] = df["log2_deviation"].abs()
    df["adjective"] = pd.Categorical(df["adjective"], ADJECTIVE_NAMES, ordered=True)
    df["scale"] = pd.Categorical(df["scale"], SCALE_NAMES, ordered=True)
    df["scale_n"] = df["scale"].map(SCALE_N).astype(float)
    return df


def _bootstrap_ci(values, statistic, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI. Returns (nan, nan) below 3 observations, where
    a resampled interval would be theatre rather than an estimate."""
    values = np.asarray([v for v in values if v == v], dtype=float)
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    stats = statistic(draws)
    tail = (1 - level) / 2
    return float(np.quantile(stats, tail)), float(np.quantile(stats, 1 - tail))


def _mean_ci(series) -> tuple[float, float]:
    return _bootstrap_ci(series, lambda d: d.mean(axis=1))


def _rate_ci(series) -> tuple[float, float]:
    """CI for a failure rate. Booleans arrive from the CSV as an object column
    once any row is blank, so coerce before resampling."""
    values = [float(v) for v in series if v == v]
    return _bootstrap_ci(values, lambda d: 1 - d.mean(axis=1))


def _with_ci(frame: pd.DataFrame, group_cols: list[str], source: pd.DataFrame,
             column: str, fn, prefix: str) -> pd.DataFrame:
    """Attach (prefix_ci_low, prefix_ci_high) to an already-aggregated frame."""
    if frame.empty or source.empty:
        frame = frame.copy()
        frame[f"{prefix}_ci_low"] = pd.Series(dtype=float)
        frame[f"{prefix}_ci_high"] = pd.Series(dtype=float)
        return frame
    bounds = (source.groupby(group_cols, observed=True)[column]
              .apply(lambda s: pd.Series(fn(s), index=["lo", "hi"]))
              .unstack())
    frame = frame.merge(bounds.rename(columns={"lo": f"{prefix}_ci_low",
                                               "hi": f"{prefix}_ci_high"}),
                        on=group_cols, how="left")
    return frame


def _style(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def noise_floor(df: pd.DataFrame) -> pd.DataFrame:
    """Within-condition spread across the 5 repeats -- the floor that any
    adjective or scale effect has to clear to mean anything."""
    scored = df[df["value"].notna()]
    per_cell = (
        scored.groupby(["recipe", "ratio", "adjective", "scale"], observed=True)
        ["log2_deviation"]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary = (
        per_cell.groupby(["recipe", "ratio"], observed=True)
        .agg(cells=("sd", "size"),
             mean_within_cell_sd=("sd", "mean"),
             between_cell_sd=("mean", "std"))
        .reset_index()
    )
    return summary


def adjective_effect(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["value"].notna()]
    keys = ["recipe", "ratio", "adjective"]
    summary = (
        scored.groupby(keys, observed=True)
        .agg(n=("abs_log2_deviation", "count"),
             mean_abs_dev=("abs_log2_deviation", "mean"),
             sd_abs_dev=("abs_log2_deviation", "std"),
             mean_signed_dev=("log2_deviation", "mean"),
             fail_rate=("ratio_passed", lambda s: 1 - s.mean()))
        .reset_index()
    )
    summary = _with_ci(summary, keys, scored, "abs_log2_deviation",
                       _mean_ci, "mean_abs_dev")
    return _with_ci(summary, keys, scored, "ratio_passed", _rate_ci, "fail_rate")


def scale_effect(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[df["value"].notna()]
    keys = ["recipe", "ratio", "scale"]
    summary = (
        scored.groupby(keys, observed=True)
        .agg(n=("abs_log2_deviation", "count"),
             mean_abs_dev=("abs_log2_deviation", "mean"),
             fail_rate=("ratio_passed", lambda s: 1 - s.mean()),
             batching_rate=("batching_detected", "mean"),
             unscorable_rate=("status", lambda s: (s != "scored").mean()))
        .reset_index()
    )
    return _with_ci(summary, keys, scored, "abs_log2_deviation",
                    _mean_ci, "mean_abs_dev")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation without pulling in scipy for one function."""
    if len(x) < 3:
        return float("nan")
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def scale_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation against serving size on a continuous axis.

    The three numeric conditions span 6 -> 173 people, which is most of a
    decade and a half, so the natural axis is log10(people) and the natural
    summary is a slope: how much |log2 deviation| is added per tenfold increase
    in the number of people served. The unmodified prompt is excluded because
    it names no number -- it is a different question, not a fourth point.
    """
    scored = df[df["value"].notna() & df["scale_n"].notna()]
    rows = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for (recipe, ratio), group in scored.groupby(["recipe", "ratio"],
                                                 observed=True):
        x = np.log10(group["scale_n"].to_numpy(dtype=float))
        y = group["abs_log2_deviation"].to_numpy(dtype=float)
        if len(x) < 3 or x.std() == 0:
            continue
        slope = float(np.polyfit(x, y, 1)[0])
        # Resample whole generations, so the CI reflects how few independent
        # recipes each slope rests on rather than how many points were plotted.
        slopes = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(len(x), size=len(x), replace=True)
            if np.std(x[idx]) == 0:
                continue
            slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
        lo, hi = ((float(np.quantile(slopes, 0.025)),
                   float(np.quantile(slopes, 0.975))) if slopes
                  else (float("nan"), float("nan")))
        rows.append({
            "recipe": recipe, "ratio": ratio, "n": len(x),
            "slope_per_decade": slope, "slope_ci_low": lo, "slope_ci_high": hi,
            "spearman_rho": _spearman(x, y),
            "crosses_zero": not (lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


def per_recipe_failure(df: pd.DataFrame) -> pd.DataFrame:
    """Failure is per generation (a recipe fails if ANY of its ratios is out of
    band), so collapse the ratio rows first.

    Deviation is NOT computed off this collapse. A generation with one scorable
    ratio and one dropped one has `recipe_passed` null, and gating deviation on
    that threw away the ratio that did score -- which mattered because the
    dropped ones are not missing at random: recipes hedge the hydration line
    ("2.5-3 cups ice-cold liquid") more at large n and under "foolproof", the
    conditions under test. So deviation is aggregated from the ratio rows
    directly, and `unscorable`/`partial` counts are reported next to it.
    """
    per_gen = (
        df.groupby(["recipe", "adjective", "scale", "hash"], observed=True)
        .agg(recipe_passed=("recipe_passed", "first"),
             mean_abs_dev=("abs_log2_deviation", "mean"),
             status=("status", "first"),
             batching_detected=("batching_detected", "first"))
        .reset_index()
    )
    scored = per_gen[per_gen["recipe_passed"].notna()]
    summary = (
        scored.groupby("recipe", observed=True)
        .agg(n_scored=("recipe_passed", "size"),
             fail_rate=("recipe_passed", lambda s: 1 - s.mean()))
        .reset_index()
    )
    summary = _with_ci(summary, ["recipe"], scored, "recipe_passed",
                       _rate_ci, "fail_rate")

    # Deviation over every scorable ratio row, independent of whether the
    # generation as a whole could be scored.
    ratio_rows = df[df["value"].notna()]
    dev = (ratio_rows.groupby("recipe", observed=True)
           .agg(n_ratio_rows=("abs_log2_deviation", "size"),
                mean_abs_dev=("abs_log2_deviation", "mean"))
           .reset_index())
    dev = _with_ci(dev, ["recipe"], ratio_rows, "abs_log2_deviation",
                   _mean_ci, "mean_abs_dev")
    summary = summary.merge(dev, on="recipe", how="outer")

    for status in ("unscorable", "partial"):
        summary[status] = summary["recipe"].map(
            per_gen[per_gen["status"] == status]
            .groupby("recipe", observed=True).size()
        ).fillna(0).astype(int)
    summary["ground_truth_basis"] = summary["recipe"].map(
        {r: GROUND_TRUTH[r]["basis"] for r in GROUND_TRUTH}
    )
    return summary, per_gen


def plot_scale_effect(df: pd.DataFrame, path) -> None:
    """Small multiples, one panel per recipe: deviation against requested
    serving size, with one line per adjective. Faceted rather than pooled
    because the four recipes' bands are not comparable."""
    scored = df[df["value"].notna()]
    recipes = [r for r in RECIPES if r in set(scored["recipe"])]
    if not recipes:
        return
    ncols = 2
    nrows = (len(recipes) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.4 * nrows),
                             sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    x = range(len(SCALE_NAMES))

    for ax, recipe in zip(axes.flat, recipes):
        _style(ax)
        subset = scored[scored["recipe"] == recipe]
        for adjective in ADJECTIVE_NAMES:
            cells = subset[subset["adjective"] == adjective].groupby(
                "scale", observed=False)["abs_log2_deviation"]
            series = cells.mean().reindex(SCALE_NAMES)
            # Bootstrap CI per point. Without it these lines invite reading a
            # 5-repeat cell mean as a real difference; most of the movement in
            # the pilot sat inside the intervals.
            bounds = {name: _mean_ci(values) for name, values in cells}
            lo = [max(series[s] - bounds.get(s, (float("nan"),) * 2)[0], 0)
                  if s in bounds else float("nan") for s in SCALE_NAMES]
            hi = [bounds.get(s, (float("nan"),) * 2)[1] - series[s]
                  if s in bounds else float("nan") for s in SCALE_NAMES]
            ax.errorbar(x, series.values, yerr=[lo, hi],
                        color=ADJ_COLOR[adjective], linewidth=2,
                        marker=ADJ_MARKER[adjective], markersize=6,
                        markeredgecolor="#fcfcfb", markeredgewidth=2,
                        elinewidth=1, capsize=3, ecolor=ADJ_COLOR[adjective],
                        label=ADJ_LABEL[adjective])
        basis = GROUND_TRUTH[recipe]["basis"]
        ax.set_title(f"{RECIPES[recipe]}  ({basis} band)", color=INK,
                     fontsize=11, loc="left")
        ax.set_xticks(list(x))
        ax.set_xticklabels([SCALE_LABEL[s] for s in SCALE_NAMES])
        ax.set_ylim(bottom=0)
    for ax in axes.flat[len(recipes):]:
        ax.set_visible(False)

    # Each panel keeps its own y scale: the four recipes' bands are on
    # different ratios and are not comparable vertically.
    for row in axes:
        row[0].set_ylabel("mean |log2 deviation|\nfrom band center",
                          color=INK_MUTED, fontsize=9)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("people the recipe was asked to serve",
                          color=INK_MUTED, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=INK_MUTED)
    fig.suptitle("Ratio drift by requested serving size", color=INK, fontsize=13,
                 x=0.02, ha="left")
    fig.text(0.02, 0.925, "bars are 95% bootstrap CIs on the cell mean",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_failure_rates(per_gen: pd.DataFrame, path) -> None:
    """Failure rate per recipe, grouped by adjective. Computed per generation
    (pie dough fails if EITHER of its ratios is out of band, which is not the
    same as averaging its two ratio-level rates). Kept as four separate groups,
    never a single pooled bar: two recipes use documented bands and two use
    inferred +/-20% bands."""
    scored = per_gen[per_gen["recipe_passed"].notna()]
    rate_by_cell = (
        scored.groupby(["recipe", "adjective"], observed=True)["recipe_passed"]
        .apply(lambda s: 1 - s.mean())
    )
    recipes = [r for r in RECIPES if r in set(scored["recipe"])]
    if not recipes:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.patch.set_facecolor("#fcfcfb")
    _style(ax)
    width = 0.26
    for offset, adjective in zip((-width, 0, width), ADJECTIVE_NAMES):
        rates = [rate_by_cell.get((r, adjective), float("nan")) for r in recipes]
        positions = [i + offset for i in range(len(recipes))]
        bars = ax.bar(positions, rates, width * 0.86, color=ADJ_COLOR[adjective],
                      label=ADJ_LABEL[adjective], edgecolor="#fcfcfb", linewidth=2)
        for bar, rate in zip(bars, rates):
            if pd.notna(rate):
                ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                        f"{rate:.0%}", ha="center", fontsize=8, color=INK_MUTED)
    ax.set_xticks(range(len(recipes)))
    ax.set_xticklabels([
        f"{RECIPES[r]}\n({GROUND_TRUTH[r]['basis']} band)" for r in recipes
    ], color=INK_MUTED)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("out-of-band rate", color=INK_MUTED, fontsize=9)
    ax.set_title("Out-of-band rate by recipe and adjective\n"
                 "documented and inferred bands are not equally strict "
                 "- compare within a recipe, not across",
                 color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def _warn_floor_ceiling(by_recipe: pd.DataFrame) -> None:
    """Say out loud when an arm has no variance left to explain.

    A recipe that passes every generation or fails every generation cannot
    show an adjective or scale effect in pass/fail terms no matter what the
    model did, and the pilot had three such arms. That is a fact about the
    band, not about the model, and it is easy to write up backwards.
    """
    if by_recipe.empty or "fail_rate" not in by_recipe:
        return
    stuck = by_recipe[(by_recipe["fail_rate"] == 0) | (by_recipe["fail_rate"] == 1)]
    if stuck.empty:
        return
    print("\n  WARNING -- floor/ceiling: "
          + ", ".join(f"{r.recipe} ({r.fail_rate:.0%} fail, n={r.n_scored})"
                      for r in stuck.itertuples()))
    print("  These arms have no pass/fail variance for an adjective or a "
          "serving size to explain. Read their mean_abs_dev instead, and treat "
          "the band itself as the thing to re-check.")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    df = load()
    RESULTS_DIR.mkdir(exist_ok=True)

    # Nothing to aggregate is a result in its own right, and almost always
    # means the extractions are stale relative to the current prompt. Say which
    # it is instead of failing somewhere deep in a groupby.
    if df.empty or df["value"].notna().sum() == 0:
        print(f"No scorable rows in {RESULTS_DIR / 'scores.csv'} "
              f"({len(df)} rows read).")
        if len(df):
            print("Status counts:")
            print(df["status"].value_counts().to_string())
            if (df["status"] == "stale_extraction").any():
                print("\nEXTRACTION_SYSTEM_PROMPT has changed since these were "
                      "extracted. Delete the affected data/*/*.extraction.json "
                      "and re-run extract.py, then score.py.")
        return

    noise = noise_floor(df)
    by_adjective = adjective_effect(df)
    by_scale = scale_effect(df)
    trend = scale_trend(df)
    by_recipe, per_gen = per_recipe_failure(df)

    noise.to_csv(RESULTS_DIR / "summary_noise_floor.csv", index=False)
    by_adjective.to_csv(RESULTS_DIR / "summary_by_adjective.csv", index=False)
    by_scale.to_csv(RESULTS_DIR / "summary_by_scale.csv", index=False)
    trend.to_csv(RESULTS_DIR / "summary_scale_trend.csv", index=False)
    by_recipe.to_csv(RESULTS_DIR / "summary_by_recipe.csv", index=False)
    per_gen.to_csv(RESULTS_DIR / "per_generation.csv", index=False)

    pd.set_option("display.width", 120, "display.max_columns", 20)
    print("\n=== Noise floor (spread within a condition, across the 5 repeats) ===")
    print(noise.to_string(index=False))
    print("\n=== Adjective effect (per recipe/ratio) ===")
    print(by_adjective.to_string(index=False))
    print("\n=== Scale effect (per recipe/ratio) ===")
    print(by_scale.to_string(index=False))
    print("\n=== Scale trend (|log2 dev| per tenfold increase in people) ===")
    if trend.empty:
        print("(not enough numeric-scale rows to fit a slope)")
    else:
        print(trend.to_string(index=False))
        print("`crosses_zero` true means the 95% bootstrap CI includes no "
              "effect -- do not describe that recipe as drifting with scale.")
    print("\n=== Per-recipe failure (do NOT pool: bands differ in kind) ===")
    print(by_recipe.to_string(index=False))
    _warn_floor_ceiling(by_recipe)
    print("\nStatus counts:")
    print(per_gen["status"].value_counts().to_string())

    plot_scale_effect(df, RESULTS_DIR / "scale_effect.png")
    plot_failure_rates(per_gen, RESULTS_DIR / "failure_rates.png")
    print(f"\nWrote summaries and plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
