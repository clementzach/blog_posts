"""Regression tests for the scoring rules, run with: python test_scoring.py

Every case here is taken from a real generation in data/ that the first version
of score.py got wrong. They are the reason the extraction schema has
option_group / is_optional, and the reason a partial bucket is not scored.
"""

from config import EXTRACTION_SCHEMA_VERSION
from score import bucket_grams, is_core, is_optional, score_one


def line(name, quantity=None, unit=None, **kw):
    item = {"name": name, "quantity": quantity, "quantity_low": None,
            "quantity_high": None, "unit": unit, "component": "recipe",
            "is_core": True, "is_optional": False, "option_group": None,
            "is_default_option": False}
    item.update(kw)
    return item


def check(label, got, want):
    status = "ok  " if got == want else "FAIL"
    print(f"  [{status}] {label}: got {got!r}, want {want!r}")
    return got == want


results = []

# panna_cotta__control__35: "45 g powdered gelatin", then a tip "use 50-55 g if
# you want to unmold", then "if using leaves, use 22-25 leaves". Summing these
# gave 156.25 g against a true 45 g and still scored as a pass.
alternatives = [
    line("powdered gelatin", 45, "g", option_group="gelatin",
         is_default_option=True),
    line("powdered gelatin", None, "g", quantity_low=50, quantity_high=55,
         option_group="gelatin"),
    line("gelatin leaves", None, "leaves", quantity_low=22, quantity_high=25,
         option_group="gelatin"),
]
bucket = bucket_grams(alternatives, "gelatin")
results.append(check("mutually exclusive alternatives collapse to the default",
                     bucket["grams"], 45.0))
results.append(check("the unchosen alternatives are reported, not dropped",
                     len(bucket["alternatives_skipped"]), 2))
results.append(check("choosing an alternative is not a partial bucket",
                     bucket["partial"], False))

# panna_cotta__avant_garde__6: "5 g powdered gelatin, or 3 gelatin leaves".
inline_or = [
    line("powdered gelatin", 5, "g", option_group="gelatin",
         is_default_option=True),
    line("gelatin leaves", 3, "gelatin leaves", option_group="gelatin"),
]
results.append(check("inline 'or' alternative is not added on top",
                     bucket_grams(inline_or, "gelatin")["grams"], 5.0))

# Two flours that genuinely both go into one dough must still be summed.
both = [line("all-purpose flour", 150, "g"), line("rye flour", 50, "g")]
results.append(check("independent lines are still summed",
                     bucket_grams(both, "flour")["grams"], 200.0))

# choux__avant_garde__default: craquelin flour marked is_core=true by the
# extractor, which pulled egg:flour from 1.47 down to 1.07.
craquelin = [
    line("plain flour", 150, "g", component="choux pastry"),
    line("plain flour", 55, "g", component="black sesame craquelin",
         is_core=True),
]
results.append(check("component denylist overrides a wrong is_core=true",
                     bucket_grams(craquelin, "flour")["grams"], 150.0))
results.append(check("craquelin component is not core",
                     is_core({"component": "black sesame craquelin",
                              "is_core": True}), False))

# pie_dough__foolproof__6: "Optional: 1 tablespoon vodka, replacing 1
# tablespoon of the water" must not be added to hydration.
optional = [
    line("ice water", None, "tablespoons", quantity_low=3, quantity_high=4),
    line("vodka", 1, "tablespoon", is_optional=True),
]
hydration = bucket_grams(optional, "water")
results.append(check("optional lines are excluded",
                     round(hydration["grams"], 2), 51.76))
results.append(check("optional line is reported",
                     hydration["optional_skipped"], ["vodka"]))
results.append(check("'optional' in the name is caught too",
                     is_optional({"name": "1 tsp sugar, optional"}), True))

# pie_dough__foolproof__35: "2.5-3 cups ice-cold liquid" is hydration; before
# the matcher fix the bucket was empty and the row was thrown away.
hedged = [line("ice-cold liquid", None, "cups", quantity_low=2.5,
               quantity_high=3)]
results.append(check("hedged hydration wording matches the water bucket",
                     bucket_grams(hedged, "water")["grams"] > 0, True))

# panna_cotta__avant_garde__default: "2 gelatin leaves" with a null unit was
# silently dropped and the remaining 4 g was scored as if complete.
dropped = [line("powdered gelatin", 4, "g"),
           line("gelatin leaves", 2, None)]
partial = bucket_grams(dropped, "gelatin")
results.append(check("a resolvable bare gelatin count is converted",
                     partial["grams"], 9.0))
truly_dropped = [line("powdered gelatin", 4, "g"),
                 line("gelatin", None, None)]
partial2 = bucket_grams(truly_dropped, "gelatin")
results.append(check("a dropped line marks the bucket partial",
                     partial2["partial"], True))

# A partial bucket must not produce a ratio.
def record_for(ingredients, version=EXTRACTION_SCHEMA_VERSION):
    return {"hash": "t", "recipe": "panna_cotta", "adjective": "control",
            "scale": "6", "repeat_index": 0,
            "extraction_schema_version": version,
            "extraction": {"batching_detected": False,
                           "ingredients": ingredients}}


record = score_one(record_for([
        line("powdered gelatin", 4, "g"),
        line("gelatin", None, None),
        line("heavy cream", 500, "g"),
]))
results.append(check("partial bucket is not scored", record["status"], "partial"))
results.append(check("partial bucket has no ratio value",
                     record["ratios"]["gelatin_to_liquid"]["value"], None))

# An extraction made before option_group existed must not be scored as if the
# alternatives had been grouped -- it would look completely normal and be wrong.
stale = score_one(record_for([line("powdered gelatin", 45, "g"),
                              line("heavy cream", 500, "g")], version=1))
results.append(check("a stale-schema extraction is refused",
                     stale["status"], "stale_extraction"))
results.append(check("a stale-schema extraction produces no ratio",
                     stale["ratios"], {}))

# "rosewater" must not enter the liquid bucket via the "water" substring.
rose = [line("heavy cream", 500, "g"),
        line("lemon juice or rosewater", 10, "g")]
results.append(check("rosewater is excluded from the liquid bucket",
                     bucket_grams(rose, "liquid")["grams"], 500.0))

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)
