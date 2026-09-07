"""Convert extracted (quantity, unit, ingredient) triples to grams.

Ratios in the ground-truth table are by weight, but free-text recipes give
volumes and counts, so every quantity has to be normalized. Densities are the
usual US baking references (King Arthur weights where they exist).
"""

GRAMS_PER_CUP = {
    "flour": 120.0,
    "cornstarch": 113.0,  # King Arthur, 4 oz/cup -- same source as the
                          # pastry-cream ground truth, so numerator and
                          # reference stay on one density.
    "butter": 227.0,
    "shortening": 205.0,
    "lard": 205.0,
    "oil": 218.0,
    "water": 236.6,
    "milk": 244.0,
    "cream": 232.0,
    "half-and-half": 242.0,
    "buttermilk": 245.0,
    "egg": 243.0,  # shelled, whisked
    "gelatin": 150.0,  # powdered
    "sugar": 200.0,
    "default": 236.6,  # water-like
}

# Volume units expressed in cups.
CUPS_PER_UNIT = {
    "cup": 1.0,
    "cups": 1.0,
    "c": 1.0,
    "tablespoon": 1 / 16,
    "tablespoons": 1 / 16,
    "tbsp": 1 / 16,
    "tbs": 1 / 16,
    "tb": 1 / 16,
    "teaspoon": 1 / 48,
    "teaspoons": 1 / 48,
    "tsp": 1 / 48,
    "ts": 1 / 48,
    "pint": 2.0,
    "pints": 2.0,
    "quart": 4.0,
    "quarts": 4.0,
    "qt": 4.0,
    "gallon": 16.0,
    "gallons": 16.0,
    "fluid ounce": 1 / 8,
    "fluid ounces": 1 / 8,
    "fl oz": 1 / 8,
    "floz": 1 / 8,
    "ml": 1 / 236.588,
    "milliliter": 1 / 236.588,
    "milliliters": 1 / 236.588,
    "millilitre": 1 / 236.588,
    "millilitres": 1 / 236.588,
    "cc": 1 / 236.588,
    "l": 1000 / 236.588,
    "liter": 1000 / 236.588,
    "liters": 1000 / 236.588,
    "litre": 1000 / 236.588,
    "litres": 1000 / 236.588,
    "dl": 100 / 236.588,
    "deciliter": 100 / 236.588,
    "deciliters": 100 / 236.588,
}

# "oz" is read as a WEIGHT ounce. US recipes use it both ways ("8 oz milk"
# usually means fluid); resolving it by ingredient type would be a coin flip,
# so it stays weight and the ambiguity is a stated limitation instead.
GRAMS_PER_WEIGHT_UNIT = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "gr": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ounces": 28.3495,
    "lb": 453.592,
    "lbs": 453.592,
    "pound": 453.592,
    "pounds": 453.592,
}

# Count-style units, resolved per canonical ingredient. Keys are (unit,
# canonical) with None meaning "any ingredient".
GRAMS_PER_COUNT = {
    ("egg", "egg"): 50.0,  # large egg, shelled
    ("large egg", "egg"): 50.0,
    ("egg white", "egg"): 33.0,
    ("egg whites", "egg"): 33.0,
    ("egg yolk", "egg"): 17.0,
    ("egg yolks", "egg"): 17.0,
    ("whole egg", "egg"): 50.0,
    ("stick", "fat"): 113.0,
    ("sticks", "fat"): 113.0,
    ("packet", "gelatin"): 7.0,
    ("packets", "gelatin"): 7.0,
    ("envelope", "gelatin"): 7.0,
    ("envelopes", "gelatin"): 7.0,
    ("sachet", "gelatin"): 7.0,
    ("sachets", "gelatin"): 7.0,
    ("sheet", "gelatin"): 2.5,
    ("sheets", "gelatin"): 2.5,
    ("leaf", "gelatin"): 2.5,
    ("leaves", "gelatin"): 2.5,
}

# Bare counts ("3 eggs", unit == null or "" or "whole") resolved by canonical.
GRAMS_PER_BARE_COUNT = {"egg": 50.0}

# A bare gelatin count is only resolvable when the NAME says which form is
# being counted. "2 gelatin" stays unconvertible: sheets and grams differ by
# a factor of ~2.5 and guessing would be false precision.
GELATIN_FORM_GRAMS = (
    ("sheet", 2.5),
    ("leaf", 2.5),
    ("leaves", 2.5),
    ("envelope", 7.0),
    ("packet", 7.0),
    ("sachet", 7.0),
)


def _density(ingredient: str, canonical: str | None) -> float:
    name = ingredient.lower()
    for key in ("half-and-half", "buttermilk", "cornstarch", "flour", "butter",
                "shortening", "lard", "oil", "cream", "milk", "water", "egg",
                "gelatin", "sugar"):
        if key in name:
            return GRAMS_PER_CUP[key]
    if canonical in GRAMS_PER_CUP:
        return GRAMS_PER_CUP[canonical]
    return GRAMS_PER_CUP["default"]


def to_grams(quantity, unit, ingredient: str, canonical: str | None = None) -> float | None:
    """Grams for one extracted ingredient line, or None if unconvertible.

    canonical is the ratio-relevant bucket the line matched ("flour", "fat",
    ...), used to resolve count units like "1 envelope" of gelatin.
    """
    if quantity is None:
        return None
    try:
        quantity = float(quantity)
    except (TypeError, ValueError):
        return None

    u = (unit or "").strip().lower().rstrip(".")
    u = u.replace("-", " ").strip()
    name = (ingredient or "").lower()

    if u in GRAMS_PER_WEIGHT_UNIT:
        return quantity * GRAMS_PER_WEIGHT_UNIT[u]
    if u in CUPS_PER_UNIT:
        return quantity * CUPS_PER_UNIT[u] * _density(ingredient, canonical)
    if (u, canonical) in GRAMS_PER_COUNT:
        return quantity * GRAMS_PER_COUNT[(u, canonical)]

    # Bare / vague count units: "3 eggs", "2 whole eggs", "4 large".
    if u in ("", "whole", "each", "count", "large", "medium", "small", "unit", "units"):
        if canonical == "egg":
            grams = GRAMS_PER_BARE_COUNT["egg"]
            if "yolk" in name:
                grams = GRAMS_PER_COUNT[("egg yolk", "egg")]
            elif "white" in name:
                grams = GRAMS_PER_COUNT[("egg white", "egg")]
            return quantity * grams
        if canonical == "gelatin":
            for form, grams in GELATIN_FORM_GRAMS:
                if form in name:
                    return quantity * grams
        return None

    # "1 egg yolk" style, where the extractor put the descriptor in the unit.
    for key, grams in GRAMS_PER_COUNT.items():
        if key[1] == canonical and key[0] in u:
            return quantity * grams

    return None
