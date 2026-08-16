import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from dataset import load_mbpp_sample
from harness import run_test_plus
from models import call_model

CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")

PERSONA_1 = (
    "You are an experienced Python developer focused on producing clean, "
    "well-structured, self-documenting code. You prefer idiomatic Python that "
    "prioritizes correctness and readability. You write in a conventional "
    "manner to ensure that code functions properly and is easy to read and debug. "
    "When applicable, you use tried and tested design patterns to ensure "
    "predictable, reliable code behavior."
)
PERSONA_2 = (
    "You are a creative, inventive Python developer focused on solving problems "
    "even when the solution requires out of the box thinking, cleverness, and novelty. "
    "Your solutions may diverge from common design patterns if needed to ensure the "
    "correct solution is implemented. You explore unconventional approaches and "
    "alternative ways of looking at problems to ensure that code functions correctly "
    "and solves the problem."
)

PERSONAS = {"persona_1": PERSONA_1, "persona_2": PERSONA_2}

ALL_MODELS = ["gpt", "gemini"]

# For each generator model, the other models in the set (excludes itself).
OTHER_MODELS = {m: [o for o in ALL_MODELS if o != m] for m in ALL_MODELS}


def reviewer_conditions(gen_model: str) -> list[tuple[str, str]]:
    """(reviewer_model, reviewer_persona) pairs for a given generator: itself
    plus every other model, each at both personas."""
    return [
        (rev_model, rev_persona)
        for rev_model in [gen_model] + OTHER_MODELS[gen_model]
        for rev_persona in PERSONAS
    ]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_cache_lock = Lock()


def _cache_key(model_label: str, system_prompt: str, user_prompt: str) -> str:
    content = f"{model_label}|{system_prompt}|{user_prompt}"
    return hashlib.sha256(content.encode()).hexdigest()


def cached_call(
    model_label: str, system_prompt: str, user_prompt: str, sleep_seconds: float = 0.0
) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(model_label, system_prompt, user_prompt)
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]

    response = call_model(model_label, system_prompt, user_prompt)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    with _cache_lock:
        if not cache_file.exists():
            cache_file.write_text(json.dumps({"model": model_label, "response": response}))
    return response


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def _extract_fenced_code(text: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_code_generation(text: str) -> tuple[str, bool]:
    """For generation: no fence is not a parse failure; use full response."""
    code = _extract_fenced_code(text)
    return (code if code else text.strip()), False


def extract_code_fix(text: str) -> tuple[str | None, bool]:
    """For fix: no fence is a parse failure."""
    code = _extract_fenced_code(text)
    if code is None:
        return None, True
    return code, False


# ---------------------------------------------------------------------------
# Review JSON parsing
# ---------------------------------------------------------------------------

def parse_review_json(text: str) -> tuple[bool | None, bool]:
    """Returns (has_bug, parse_failure)."""
    try:
        data = json.loads(text.strip())
        return bool(data["has_bug"]), False
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    match = re.search(r'\{[^}]*"has_bug"\s*:\s*(true|false)[^}]*\}', text, re.IGNORECASE)
    if match:
        try:
            data = json.loads(match.group(0))
            return bool(data["has_bug"]), False
        except (json.JSONDecodeError, KeyError):
            pass

    return None, True


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM = "{persona}"

_GENERATION_USER = (
    "Write a Python function to solve the following problem:\n\n"
    "{problem_text}\n\n"
    "Your function will be called like this:\n"
    "{example_test}\n\n"
    "Return only the code inside a ```python ... ``` fenced block."
)

_REVIEW_SYSTEM = (
    "{persona}\n\n"
    "Review Python code for correctness (defined as whether the code will fully solve all parts "
    "of the problem statement without any errors). If you see something that might be a possible "
    "bug, err on the side of reporting a bug rather than letting one slip by. "
    "When asked, respond ONLY with valid JSON "
    'in this exact format: {{"has_bug": true}} or {{"has_bug": false}}. No other text.'
)

_REVIEW_USER = (
    "Problem statement:\n{problem_text}\n\n"
    "Code to review:\n```python\n{code}\n```\n\n"
    "Does this code seem likely to contain a bug that would cause it to not function correctly "
    'for the given problem? Respond with JSON only: {{"has_bug": true}} or {{"has_bug": false}}. '

)

_FIX_SYSTEM = (
    "{persona}\n\n"
    "You are fixing buggy Python code. Return the corrected code inside a "
    "```python ... ``` fenced block and nothing else."
)

_FIX_USER = (
    "Problem statement:\n{problem_text}\n\n"
    "Buggy code:\n```python\n{code}\n```\n\n"
    "Provide a corrected version inside a ```python ... ``` fenced block."
)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _example_test(problem: dict) -> str:
    """First test case, revealed in the generation prompt to disambiguate
    argument order / return shape / parameter count."""
    return problem["test_list"][0] if problem["test_list"] else ""


def generate_code(model: str, problem: dict, sleep_seconds: float = 0.0) -> tuple[str, bool]:
    system = PERSONA_1
    user = _GENERATION_USER.format(
        problem_text=problem["prompt"], example_test=_example_test(problem)
    )
    response = cached_call(model, system, user, sleep_seconds)
    return extract_code_generation(response)


def review_code(
    reviewer_model: str,
    reviewer_persona_key: str,
    problem: dict,
    code: str,
    sleep_seconds: float = 0.0,
) -> tuple[bool | None, bool]:
    persona = PERSONAS[reviewer_persona_key]
    system = _REVIEW_SYSTEM.format(persona=persona)
    user = _REVIEW_USER.format(problem_text=problem["prompt"], code=code)
    response = cached_call(reviewer_model, system, user, sleep_seconds)
    return parse_review_json(response)


def get_fix(
    reviewer_model: str,
    reviewer_persona_key: str,
    problem: dict,
    code: str,
    sleep_seconds: float = 0.0,
) -> tuple[str | None, bool]:
    persona = PERSONAS[reviewer_persona_key]
    system = _FIX_SYSTEM.format(persona=persona)
    user = _FIX_USER.format(problem_text=problem["prompt"], code=code)
    response = cached_call(reviewer_model, system, user, sleep_seconds)
    return extract_code_fix(response)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    n_problems: int | None = None,
    sleep_seconds: float = 0.0,
    max_workers: int = 100,
    use_batch: bool = True,
) -> list[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)
    problems = load_mbpp_sample(n_problems)

    if use_batch:
        import batch

        print(f"Batch precache: generation ({len(ALL_MODELS)} models × {len(problems)} problems)")
        for model in ALL_MODELS:
            calls = [
                (
                    PERSONA_1,
                    _GENERATION_USER.format(
                        problem_text=problem["prompt"], example_test=_example_test(problem)
                    ),
                )
                for problem in problems
            ]
            batch.batch_precache(model, calls)

    # Step 1: Generation (all model × problem pairs in parallel)
    print(f"\nStep 1: Generating code ({len(ALL_MODELS)} models × {len(problems)} problems)...")
    generations: dict[tuple[str, int], dict] = {}
    total_gen = len(ALL_MODELS) * len(problems)
    gen_count = 0
    gen_lock = Lock()

    def _gen_task(model: str, problem: dict) -> tuple[tuple[str, int], dict]:
        code, parse_fail = generate_code(model, problem, sleep_seconds)
        return (model, problem["task_id"]), {"code": code, "parse_failure_generation": parse_fail}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_gen_task, model, problem): (model, problem["task_id"])
            for model in ALL_MODELS
            for problem in problems
        }
        for future in as_completed(futures):
            key, data = future.result()
            generations[key] = data
            with gen_lock:
                gen_count += 1
                print(f"  [{gen_count}/{total_gen}] gen={key[0]} problem={key[1]}", end="\r")
    print()

    # Step 2: Ground truth labeling (subprocess per pair, all independent)
    print("Step 2: Labeling ground truth...")
    ground_truth: dict[tuple[str, int], bool] = {}
    gt_lock = Lock()

    def _gt_task(model: str, problem: dict) -> tuple[tuple[str, int], tuple[bool, str | None]]:
        code = generations[(model, problem["task_id"])]["code"]
        passed, error = run_test_plus(code, problem["test"], problem["test_imports"])
        return (model, problem["task_id"]), (passed, error)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        gt_futures = [executor.submit(_gt_task, model, problem)
                      for model in ALL_MODELS for problem in problems]
        for future in as_completed(gt_futures):
            key, result = future.result()
            with gt_lock:
                ground_truth[key] = result

    if use_batch:
        import batch

        print("Batch precache: review calls")
        review_calls: dict[str, list[tuple[str, str]]] = {m: [] for m in ALL_MODELS}
        for gen_model in ALL_MODELS:
            for rev_model, rev_persona in reviewer_conditions(gen_model):
                system = _REVIEW_SYSTEM.format(persona=PERSONAS[rev_persona])
                for problem in problems:
                    code = generations[(gen_model, problem["task_id"])]["code"]
                    user = _REVIEW_USER.format(problem_text=problem["prompt"], code=code)
                    review_calls[rev_model].append((system, user))
        for model in ALL_MODELS:
            batch.batch_precache(model, review_calls[model])

    # Step 3 + 4: Review and fix (all combinations in parallel; fix is sequential within each)
    # NOTE: get_fix calls are not batch-precached -- whether a fix is needed
    # depends on the review outcome above, so those calls stay live.
    total_reviews = len(ALL_MODELS) * len(ALL_MODELS) * len(PERSONAS) * len(problems)
    print(f"Step 3+4: Reviewing ({total_reviews} calls)...")
    results: list[dict] = []
    review_count = 0
    results_lock = Lock()

    def _review_fix_task(
        gen_model: str, rev_model: str, rev_persona: str, problem: dict
    ) -> dict:
        task_id = problem["task_id"]
        gen_data = generations[(gen_model, task_id)]
        code = gen_data["code"]
        pre_review_pass, pre_review_error = ground_truth[(gen_model, task_id)]

        has_bug, parse_failure_review = review_code(
            rev_model, rev_persona, problem, code, sleep_seconds
        )

        corrected_code = None
        post_review_pass = None
        parse_failure_regeneration = False

        if has_bug is True:
            fixed_code, parse_failure_regeneration = get_fix(
                rev_model, rev_persona, problem, code, sleep_seconds
            )
            corrected_code = fixed_code
            if fixed_code:
                post_review_pass, _ = run_test_plus(
                    fixed_code, problem["test"], problem["test_imports"]
                )

        return {
            "problem_id": task_id,
            "generator_model": gen_model,
            "generator_persona": "persona_1",
            "reviewer_model": rev_model,
            "reviewer_persona": rev_persona,
            "bug_detected": has_bug,
            "pre_review_code": code,
            "corrected_code": corrected_code,
            "successful_run_pre_review": pre_review_pass,
            "pre_review_error": pre_review_error,
            "successful_run_post_review": post_review_pass,
            "parse_failure_generation": gen_data["parse_failure_generation"],
            "parse_failure_review": parse_failure_review,
            "parse_failure_regeneration": parse_failure_regeneration,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rf_futures = {}
        for gen_model in ALL_MODELS:
            for rev_model, rev_persona in reviewer_conditions(gen_model):
                for problem in problems:
                    f = executor.submit(_review_fix_task, gen_model, rev_model, rev_persona, problem)
                    rf_futures[f] = (gen_model, rev_model, rev_persona, problem["task_id"])

        for future in as_completed(rf_futures):
            record = future.result()
            with results_lock:
                results.append(record)
                review_count += 1
                gen_model, rev_model, rev_persona, task_id = rf_futures[future]
                print(
                    f"  [{review_count}/{total_reviews}] "
                    f"gen={gen_model} rev={rev_model}/{rev_persona} prob={task_id}",
                    end="\r",
                )

    print()
    return results
