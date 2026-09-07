# LLM Recipe Ratio-Fidelity Evaluation

Does a chat model keep the *functionally load-bearing* ingredient ratios stable
when you vary the adjective in the request and the serving size you ask for —
including serving sizes far outside anything a published recipe targets?

This tests **free-text generation as an ordinary chat user receives it**: no
system prompt, no tool calling, no JSON mode, temperature and reasoning effort
left at their defaults, no seed. Structured-output elicitation is a different
question and is out of scope here.

## Grid

4 recipes × 3 adjectives × 4 scale conditions = 48 prompts × 5 repeats =
**240 generations**, plus one extraction call each (240). Estimated cost at
current pricing (\$0.20/1M in, \$1.20/1M out): **~\$0.25–0.30**.

| Axis | Levels |
|---|---|
| Recipe | pie dough (as "pie crust"), panna cotta, pastry cream, choux pastry |
| Adjective | "really tasty" (control), "foolproof" (safety-pulling), "avant garde" (creativity-pulling) |
| Scale | no modifier, "for 6 people", "for 35 people", "for 173 people" |

Wording lives in `config.json`, not in call logic. **Lock it before the full
batch** — changing it invalidates every cached generation.

## Ground truth (`config.json` → `ground_truth`)

| Recipe | Ratio | Pass band | Basis |
|---|---|---|---|
| Pie dough | fat:flour by weight | 0.68–0.89 | Convergence across 8 independently-tested recipes |
| | water:flour by weight | 0.28–0.50 | Convergence across 8 independently-tested recipes |
| Panna cotta | gelatin:liquid, weight % | 0.65%–3.9% | Documented range across tested/published recipes (set/don't-set language) |
| Pastry cream | cornstarch:milk, weight % | 4.1%–8.3% | Convergence across 12 independently-tested recipes (King Arthur x2, Scotch & Scones, Sally's Baking Addiction, Allrecipes, Serious Eats, Preppy Kitchen, Martha Stewart, Stay at Home Chef, Natasha's Kitchen, Food Network, one additional source); dense cluster ~4.8%–6.6%, with King Arthur GF (4.1%) and Preppy Kitchen's richer 6-yolk version (8.3%) as outer bounds |
| Choux | egg:flour by weight | 1.3:1–2.3:1 | Convergence across 10 independently-tested recipes (Ruhlman canonical, America's Test Kitchen, Food Network, Sally's Baking Addiction, Allrecipes, Bonni Bakery, King Arthur, The Flavor Bender, The Buttery Crust, one additional source) |

Pie dough fails if **either** of its two ratios is out of band, not the average.

`analyze.py` prints a floor/ceiling warning whenever an arm hits 0% or 100%.

**All four ground-truth bands are now documented ranges**, not a single tested
ratio ± a fixed tolerance: pastry cream and choux were re-derived from
convergence across multiple independently-tested recipes rather than one
source ± 20%. `score.py` records `ground_truth_basis` on every record and
`analyze.py` refuses to pool failure
rates across the four.

## Pipeline

```bash
./setup.sh                        # venv + deps, copies .env.example -> .env
source .venv/bin/activate

python test_scoring.py            # 0. scoring regression tests (no API, no data)
python generate.py                # 1. 240 generations, no system prompt
python extract.py                 # 2. 240 extraction calls (1:1)
python validate_extraction.py     # 3. build the hand-check worksheet
python validate_extraction.py --report
python score.py                   # 4. ratios, pass/fail, deviation magnitude
python analyze.py                 # 5. summaries + plots
```

`generate.py` and `extract.py` take `--limit N` and `--workers N`; a smoke test
is `python generate.py --repeats 1 --limit 4`.

### Stage notes

- **generate.py** — one user message, no system prompt, no temperature, no seed.
- **extract.py** — separate call per generation, same model, with an extraction
  system prompt that asks for name/quantity/unit as JSON, requires `null` for
  ambiguous or vague quantities rather than a guessed number, flags multi-batch
  recipes (with per-batch quantities and a batch count), marks each line
  `is_core`/`is_optional`, and groups mutually exclusive alternatives under an
  `option_group` with one `is_default_option`. Every record is stamped with
  `EXTRACTION_SCHEMA_VERSION` (`config.py`); bump it whenever a prompt change
  alters what `score.py` can rely on.
- **validate_extraction.py** — stratified sample of 28 generations rendered as
  a markdown worksheet showing the recipe text, the extractor's JSON, **and the
  bucket totals `score.py` derives from it**, plus a CSV with three judgment
  columns: `missed_or_misread`, `false_precision` (extractor resolved an
  ambiguous quantity into a false-precise number), and `wrong_line_semantics`
  (numbers right, but a line grouped or classified wrong — see items 2–3
  under *Extraction problems* below). The bucket totals are printed because the third category is invisible
  in the JSON: the only way to see that three gelatin lines were summed is to
  see the gelatin bucket read 156 g against a recipe that says 45 g. Mark the
  `reviewed` column on each row you check — rates are computed over reviewed
  rows only. Accept threshold, fixed in advance: **all three rates ≤ 10%**.
  Above that, revise `EXTRACTION_SYSTEM_PROMPT`, bump
  `EXTRACTION_SCHEMA_VERSION` in `config.py`, delete the affected
  `*.extraction.json`, re-run.
- **score.py** — pure local computation, always safe to re-run. Converts every
  quantity to grams (`units.py`), matches ratio-relevant ingredients by partial
  string match (`config.json` → `ingredient_matchers`), and records the
  continuous deviation (`deviation_from_center`, `log2_deviation`,
  `outside_band_by`) alongside the binary pass/fail. Three rules govern which
  matched lines a bucket may add up — alternatives are chosen not summed,
  non-core lines are excluded with the extractor overridable, and a partially
  dropped bucket is not scored at all. Refuses to score any extraction stamped
  with an older `EXTRACTION_SCHEMA_VERSION` (status `stale_extraction`).
- **analyze.py** — noise floor (within-condition spread across the 5 repeats),
  adjective effect, scale effect, and failure rate reported per recipe. Scale
  is treated ordinally as promised: `summary_scale_trend.csv` fits
  |log2 deviation| against log10(people) across the three numeric conditions
  and reports a slope per decade with a bootstrap CI and a `crosses_zero` flag.
  Every rate and mean carries a 95% bootstrap CI, and the scale plot carries
  error bars — at 5 repeats a cell proportion of 0/5 is not evidence of much.
  **Deviation is the primary outcome; pass/fail is secondary** (see the
  floor/ceiling box above). Writes `results/summary_*.csv` and two PNGs.
- **test_scoring.py** — regression tests, one per bug found in the pilot, each
  built from the real generation that exposed it. No API key or cached data
  needed. Run it after touching `score.py`, `units.py`, or the matchers.

## Storage

```
data/{recipe}__{adjective}__{scale}/
    manifest.csv                 repeat_index -> hash, status
    {hash}.generation.json       prompt, raw response, model, api params, timestamp
    {hash}.extraction.json       ingredients, batching flag, extractor raw output
    {hash}.score.json            ratios, pass/fail, deviation magnitude
```

`hash` is a UUID4 assigned at generation time, **not** a content hash: repeats
are intentionally stochastic, so a content hash would be meaningless across
them. Every stage checks for its own output before calling the API, so a crash
at generation 150/240 resumes at 151 without re-spending on completed calls.
Scoring is local and always overwrites.

## Extraction problems found in the pilot, and how they are handled

Each of these silently produced a plausible-looking wrong number rather than an
error, which is why every one of them has a test in `test_scoring.py`:

1. **Ranged quantities.** Pie dough states its water as "6–8 tablespoons"
   nearly every time. The extractor correctly refuses to guess a number, which
   left water:flour uncomputable and made most pie-dough generations
   unscorable. The extractor now also reports `quantity_low`/`quantity_high`
   for *explicitly stated numeric ranges* (`quantity` stays `null`), and
   `score.py` scores the midpoint with `used_range_midpoint` recorded on the
   row. Vague amounts ("a few", "to taste") still have no endpoints and stay
   unscorable — the no-false-precision rule is intact.
2. **Non-core ingredient lines.** An egg wash of "1 beaten egg + 1 tbsp water
   or milk" leaked into the water bucket and produced a plausible-looking but
   wrong water:flour of 0.048. The extractor now labels each line with a
   `component` and an `is_core` boolean, and `score.py` counts only core lines.
   This matters most for choux (egg wash) and pie dough.

   The extractor does **not** get the last word on this. It marked a black
   sesame craquelin's 55 g of flour as `is_core: true`, which merged into the
   choux flour and moved that generation's egg:flour from 1.47 to 1.07 — the
   most extreme value in the whole choux arm, and an artifact. `is_core` and
   the component denylist are now ANDed. Note the direction of the bias: the
   avant-garde arm generates the most extra components, so this error was
   correlated with the independent variable.

3. **Mutually exclusive alternatives were summed instead of chosen.** The worst
   of the three. Recipes routinely state one amount several ways — "45 g
   powdered gelatin", then a tip "use 50–55 g if you want to unmold it", then
   "if using leaves, use 22–25 leaves". All three matched the gelatin bucket
   and were added together:

   ```
   numerator_grams = 156.25   # 45 + 52.5 + 58.75
   true value      = 45
   ```

   Three of four panna cotta conditions had a ~3× inflated numerator and
   **still scored `passed: true`**, because the 0.65–3.9% band is wide enough
   to swallow the error. The extractor now assigns an `option_group` to lines
   the cook chooses between and flags one `is_default_option`; `score.py`
   collapses each group to one line and reports the rest in
   `alternatives_skipped`. Genuine multi-line contributions (bread flour *and*
   rye flour in one dough) have no group and are still summed.

4. **Partially dropped buckets were scored as if complete.** A bucket only
   counted as unscorable when it totalled zero, so a bucket that lost one line
   to a null quantity or an unconvertible unit reported a real number missing
   part of itself. Those rows now get status `partial` and no ratio.

5. **Non-random missingness in the hydration bucket.** "2½–3 cups ice-cold
   liquid" and "half ice water and half cold vodka" matched nothing, so the row
   was thrown away — and recipes hedge the hydration line more at large n and
   under "foolproof", exactly the conditions under test. The `water` matcher now
   covers those wordings. `analyze.py` also no longer gates deviation on
   `recipe_passed`, which was discarding a perfectly good `fat_to_flour` every
   time its partner ratio dropped out.

## Batching

`batching_detected` is recorded on every score row but **never affects
pass/fail**: a ratio holds per batch and in total, so batching changes only
absolute quantities. At n=173, splitting into batches is often the *correct*
answer, not a failure — it is a variable to analyze, not a penalty.

## Known limitations (state these in any writeup)

- Free-text elicitation tests what a normal chat user receives, not
  tool-calling/JSON-mode reliability — a separate question, out of scope.
- All four ground-truth bands are documented ranges, but they draw on
  different numbers of independent sources (pie dough and choux each converge
  across several tested recipes; panna cotta and pastry cream lean on fewer).
  Not necessarily equally strict; do not pool without checking.
- Pie dough's fat:flour band is an industry *quality* spec, not the literal
  structural failure point (~130%+). A small overage is scored as a failure
  even though the dough might still physically hold together.
- Ingredient matching is partial-string only at v1. Full synonym mapping
  (salted vs. unsalted butter, cream vs. half-and-half) is deferred to analysis
  once the data shows which variants actually appear.
- Pastry cream's denominator is total dairy liquid (`dairy_liquid`), not the
  milk line alone: it is what the cornstarch has to set. The King Arthur
  reference is all milk, so the two agree there, but a generation that swaps in
  cream is measured on the larger quantity. Say "cornstarch:dairy", not
  "cornstarch:milk".
- Choosing one of several stated alternatives means the score reflects the
  recipe's *default* path, not the range of paths it offers. A recipe that
  offers a soft set and a firm set is scored on the one in its ingredient list.
- Recipes that specify an ingredient by consistency rather than amount are
  scored on the stated amount, which is an upper bound. Nearly every choux
  generation says "add the egg gradually — you may not need all of it", so
  egg:flour is measured against the maximum egg, not the egg a cook would use.
  Choux arguably does not specify this ratio at all; it specifies an endpoint.
  This is a real limit on what the choux arm can mean and belongs in the
  writeup, not just here.
- "oz" is read as a weight ounce. US recipes use it both ways ("8 oz milk"
  usually means fluid); resolving it by ingredient type would be a coin flip.
- Volume-to-weight conversion uses standard US baking densities; a recipe using
  unusual measures will convert approximately.
- Panna cotta's "liquid" denominator includes gelatin-blooming water along with
  the dairy. At typical amounts this shifts the ratio negligibly.
- N=5 repeats gives a rough, not precise, noise-floor estimate. Every summary
  carries a bootstrap CI for this reason; at n=5 most cell-to-cell differences
  in the pilot sat inside their intervals.
- Mayonnaise was considered and dropped: its oil:yolk ratio is not the actual
  constraint on emulsion success (water:oil ratio and technique are), so
  ratio-based scoring would not have measured anything real.
