"""Code-tokens-only perplexity scoring with Qwen2.5-Coder-7B (base).

The code chunk is embedded in a review-style prompt prefix/suffix (mirroring
``pipeline._REVIEW_USER``). The prefix and suffix tokens are provided as
context but masked out of the loss (``labels = -100``); only the tokens of the
code chunk contribute to the NLL. See STUDY_DESIGN.md section 5.2.

Two metrics per (chunk, transform):
* ``perplexity_raw``        = exp(mean NLL over code tokens)
* ``perplexity_normalized`` = mean NLL / ln(2) / num_chars   (bits/char)

Results are cached under ``sha256("qwen_ppl|" + code)`` in the shared cache dir.
"""

import hashlib
import json
import math
from pathlib import Path
from threading import Lock

from pipeline import PERSONA_1, _REVIEW_SYSTEM

CACHE_DIR = Path("cache")
MODEL_ID = "Qwen/Qwen2.5-Coder-7B"

_system = _REVIEW_SYSTEM.format(persona=PERSONA_1)

_qwen_base: dict = {}
_cache_lock = Lock()


def _load_qwen_base():
    if _qwen_base:
        return _qwen_base["model"], _qwen_base["tokenizer"], _qwen_base["device"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto").to(device)
    model.eval()
    _qwen_base.update(model=model, tokenizer=tokenizer, device=device)
    return model, tokenizer, device


def free_model() -> None:
    """Release the base model's weights to reclaim RAM.

    Phase 1 runs perplexity (base model) and detection (instruct model) on a
    16 GB CPU-only budget; the two 7B models cannot be resident at once. The
    pipeline calls this after the perplexity pass, before the instruct model
    loads. See STUDY_DESIGN.md section 5.
    """
    import gc

    _qwen_base.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cache_key(code: str) -> str:
    return hashlib.sha256(f"qwen_ppl|{code}".encode()).hexdigest()


def _build_context(problem_text: str, code: str) -> tuple[str, str]:
    """Return (prefix, suffix) text surrounding the code chunk."""
    prefix = (
        f"{_system}\n\n"
        f"Problem statement:\n{problem_text}\n\n"
        f"Code to review:\n```python\n"
    )
    suffix = "\n```\n"
    return prefix, suffix


def score_perplexity(problem_text: str, code: str) -> dict:
    """Compute (and cache) perplexity metrics for one code chunk."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(code)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    import torch

    model, tokenizer, device = _load_qwen_base()
    prefix, suffix = _build_context(problem_text, code)

    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    code_ids = tokenizer(code, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]

    input_ids = torch.tensor([prefix_ids + code_ids + suffix_ids], device=device)
    labels = input_ids.clone()
    # Mask everything except the code tokens.
    labels[0, : len(prefix_ids)] = -100
    labels[0, len(prefix_ids) + len(code_ids) :] = -100

    with torch.no_grad():
        logits = model(input_ids).logits

    # Causal shift: token t is predicted from logits at t-1.
    shift_logits = logits[0, :-1, :]
    shift_labels = labels[0, 1:]
    mask = shift_labels != -100
    n_code_tokens = int(mask.sum())

    log_probs = torch.log_softmax(shift_logits[mask].float(), dim=-1)
    token_nll = -log_probs[torch.arange(n_code_tokens), shift_labels[mask]]
    mean_nll = float(token_nll.mean())

    num_chars = max(len(code), 1)
    result = {
        "perplexity_raw": math.exp(mean_nll),
        "perplexity_normalized": mean_nll / math.log(2) / num_chars,
        "mean_nll": mean_nll,
        "n_code_tokens": n_code_tokens,
        "num_chars": num_chars,
    }

    with _cache_lock:
        if not cache_file.exists():
            cache_file.write_text(json.dumps(result))
    return result
