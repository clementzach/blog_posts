"""Model calls for generation and extraction.

Generation mimics the default consumer chat experience: standard Chat
Completions, no system prompt, temperature unset, no seed, default
reasoning effort. Do not "clean up" those defaults -- they are the condition
under test.
"""

import json
import os
import re

from openai import OpenAI

MODEL = os.environ.get("RECIPE_MODEL", "gpt-5.6-luna")

EXTRACTION_SYSTEM_PROMPT = """\
You extract structured ingredient data from recipe text. Return ONLY a JSON \
object, no prose and no code fence, with this shape:

{
  "ingredients": [
    {"name": "<ingredient as written, lowercase>",
     "quantity": <number or null>,
     "quantity_low": <number or null>,
     "quantity_high": <number or null>,
     "unit": "<unit as written, or null>",
     "component": "<which part of the recipe, e.g. dough, filling, egg wash>",
     "is_core": <true or false>,
     "is_optional": <true or false>,
     "option_group": "<group id, or null>",
     "is_default_option": <true or false>}
  ],
  "batching_detected": <true or false>,
  "batch_note": "<quote the batching language, or null>",
  "per_batch_ingredients": [ {"name": ..., "quantity": ..., "unit": ...} ],
  "batch_count": <number or null>
}

Rules:
- Quantities must be numbers (convert fractions and mixed numbers: "1 1/2" -> \
1.5, "3/4" -> 0.75). Ranges, vague amounts, and unstated amounts ("a few" \
"to taste", "as needed") MUST have quantity null. Do not \
guess a specific number, and do not pick the midpoint or an endpoint of a range.
- If, and only if, the recipe states an explicit numeric range ("6-8 \
tablespoons"), also fill in "quantity_low" and "quantity_high" with the stated \
endpoints (quantity itself stays null). Otherwise both are null. Never invent \
endpoints for a vague amount like "a few" or "to taste".
- ALWAYS give a "unit". For a bare count, use the counted noun itself \
("eggs", "sheets", "leaves", "envelopes"). Never leave the unit null when a \
quantity is present.
- "component" names the part of the recipe the ingredient belongs to, in the \
recipe's own words ("dough", "crust", "custard", "filling", "egg wash", \
"glaze", "craquelin", "garnish", "topping", "sauce", "serving"). If the recipe \
has only one component, use the recipe name.
- "is_core" is true when the ingredient goes into the main preparation the \
prompt asked for, and false when it belongs to a wash, glaze, garnish, \
topping, crumble, craquelin, streusel, filling, sauce, dusting, or serving \
suggestion. An egg wash brushed on a crust is NOT core. A craquelin disc on \
top of a choux bun is NOT core, even though it is made of the same flour and \
butter as the pastry.
- "is_optional" is true when the recipe presents the ingredient as skippable \
("optional", "if you like", "for a richer version"). An optional ingredient \
is still listed.

MUTUALLY EXCLUSIVE ALTERNATIVES -- read this carefully, it is the part that is \
most often got wrong:
- When the recipe offers two or more amounts or forms of the SAME ingredient \
and the cook picks ONE of them, every one of those lines gets the SAME \
"option_group" string, and exactly one of them gets "is_default_option": true.
- Triggers include the word "or" ("5 g powdered gelatin, or 3 gelatin \
leaves"), and any tip or variation that restates an amount for a different \
outcome ("use 50-55 g if you want to unmold it", "for a softer set, use \
210-220 g", "if using leaves, use 22-25 leaves"). These are alternatives to \
the main ingredient line, NOT additional quantities to be added on top of it.
- The default option is the one in the recipe's own ingredient list -- the \
amount a cook following the recipe straight through would use. Variations \
offered in a tips or notes section are never the default.
- Name the group after the ingredient, e.g. "gelatin", "hydration", "fat".
- An ingredient that appears once, or that is genuinely added IN ADDITION to \
another line (bread flour AND rye flour both going into one dough), has \
"option_group": null and "is_default_option": false. Only use a group when \
the cook is choosing between the lines.
- Keep the unit exactly as the recipe states it ("cups", "g", "envelope", \
"large", "eggs").
- If an ingredient is listed with both volume and weight, prefer the weight. \
That is one line, not two alternatives.
- Set "batching_detected" true only if the recipe text says to make the recipe \
in multiple separate batches (e.g. "divide into two batches", "make in 3 \
batches of X"). When it is true, put the per-batch quantities in \
"per_batch_ingredients" and the number of batches in "batch_count". When it is \
false, "per_batch_ingredients" must be [] and "batch_count" null.
- "ingredients" is always the TOTAL quantity for the whole recipe as written, \
and lists every ingredient, core or not. If the recipe gives both a total and \
a per-batch breakdown, "ingredients" holds only the totals -- do not also add \
the per-batch lines to it.
"""


def _client() -> OpenAI:
    return OpenAI()


def generate_recipe(prompt: str) -> dict:
    """One generation call. Returns the response text plus the api params used."""
    params = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = _client().chat.completions.create(**params)
    return {
        "text": response.choices[0].message.content,
        "api_params": params,
        "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def extract_ingredients(recipe_text: str) -> dict:
    """One extraction call. Returns parsed JSON plus the extractor's raw text."""
    params = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": recipe_text},
        ],
    }
    response = _client().chat.completions.create(**params)
    raw = response.choices[0].message.content
    return {
        "raw": raw,
        "parsed": parse_json(raw),
        "api_params": params,
        "response_id": response.id,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def parse_json(text: str | None) -> dict | None:
    """Parse the extractor's output, tolerating a stray code fence."""
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
