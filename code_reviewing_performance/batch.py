"""Batch-API precaching for gpt and gemini calls.

Submits every not-yet-cached (model, system_prompt, user_prompt) call as a
single provider batch job instead of one live request per call, at the
provider's batch discount (~50% off, hours of turnaround instead of seconds).
Results are written straight into the existing cache/{key}.json files
(``pipeline._cache_key`` format), so a subsequent ``pipeline.cached_call`` for
any of those calls is a cache hit and never makes a live request.

Only gpt and gemini are supported -- the only two models in this study.
"""

import json
import time

from google import genai
from google.genai import types
from openai import OpenAI

from models import MODEL_IDS
from pipeline import CACHE_DIR, _cache_key

POLL_INTERVAL_SECONDS = 30

_ENDED_OPENAI_STATUSES = {"completed", "failed", "expired", "cancelled"}
_ENDED_GEMINI_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}


def _uncached_calls(model_label: str, calls: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    CACHE_DIR.mkdir(exist_ok=True)
    pending = []
    seen_keys = set()
    for system_prompt, user_prompt in calls:
        key = _cache_key(model_label, system_prompt, user_prompt)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if not (CACHE_DIR / f"{key}.json").exists():
            pending.append((key, system_prompt, user_prompt))
    return pending


def _write_cache(model_label: str, key: str, response: str) -> None:
    cache_file = CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        cache_file.write_text(json.dumps({"model": model_label, "response": response}))


def batch_precache(
    model_label: str, calls: list[tuple[str, str]], poll_interval: float = POLL_INTERVAL_SECONDS
) -> None:
    """Submit every uncached (system_prompt, user_prompt) pair in `calls` as one
    batch job for `model_label`, wait for it, and write results to cache/.

    After this returns, ``pipeline.cached_call(model_label, ...)`` for any call
    in `calls` is a cache hit -- no live API call.
    """
    pending = _uncached_calls(model_label, calls)
    if not pending:
        print(f"  batch precache [{model_label}]: nothing to do ({len(calls)} calls already cached)")
        return
    print(
        f"  batch precache [{model_label}]: submitting {len(pending)} new calls "
        f"(of {len(calls)} requested)"
    )

    if model_label == "gpt":
        _run_openai_batch(pending, poll_interval)
    elif model_label == "gemini":
        _run_gemini_batch(pending, poll_interval)
    else:
        raise ValueError(f"No batch support for model_label={model_label!r}")


# ---------------------------------------------------------------------------
# OpenAI Batch API
# ---------------------------------------------------------------------------

def _run_openai_batch(pending: list[tuple[str, str, str]], poll_interval: float) -> None:
    client = OpenAI()

    lines = [
        json.dumps({
            "custom_id": key,
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL_IDS["gpt"],
                "reasoning": {"effort": "medium"},
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        })
        for key, system_prompt, user_prompt in pending
    ]
    batch_input_path = CACHE_DIR / "_openai_batch_input.jsonl"
    batch_input_path.write_text("\n".join(lines))

    with batch_input_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"    submitted OpenAI batch {batch.id}")

    while batch.status not in _ENDED_OPENAI_STATUSES:
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        print(f"    OpenAI batch {batch.id}: {batch.status}", end="\r")
    print()

    if batch.status != "completed":
        raise RuntimeError(f"OpenAI batch {batch.id} ended with status {batch.status!r}")

    output_lines = client.files.content(batch.output_file_id).text.splitlines()
    for line in output_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        key = record["custom_id"]
        if record.get("error"):
            print(f"    WARNING: OpenAI batch request {key} errored: {record['error']}")
            continue
        text = _extract_openai_text(record["response"]["body"])
        _write_cache("gpt", key, text)


def _extract_openai_text(body: dict) -> str:
    if body.get("output_text"):
        return body["output_text"]
    for item in body.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "")
    return ""


# ---------------------------------------------------------------------------
# Gemini Batch API
# ---------------------------------------------------------------------------

def _run_gemini_batch(pending: list[tuple[str, str, str]], poll_interval: float) -> None:
    client = genai.Client()

    inline_requests = [
        types.InlinedRequest(
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            ),
        )
        for _, system_prompt, user_prompt in pending
    ]
    batch_job = client.batches.create(model=MODEL_IDS["gemini"], src=inline_requests)
    print(f"    submitted Gemini batch {batch_job.name}")

    while batch_job.state.name not in _ENDED_GEMINI_STATES:
        time.sleep(poll_interval)
        batch_job = client.batches.get(name=batch_job.name)
        print(f"    Gemini batch {batch_job.name}: {batch_job.state.name}", end="\r")
    print()

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Gemini batch {batch_job.name} ended with state {batch_job.state.name}")

    # inlined_responses are returned in the same order as the input requests.
    responses = batch_job.dest.inlined_responses or []
    for (key, _, _), inline_response in zip(pending, responses):
        if inline_response.error:
            print(f"    WARNING: Gemini batch request {key} errored: {inline_response.error}")
            continue
        text = inline_response.response.text if inline_response.response else ""
        _write_cache("gemini", key, text)
