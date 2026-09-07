"""Resumable, idempotent on-disk cache.

data/{recipe}__{adjective}__{scale}/
    manifest.csv                  repeat_index -> hash, status
    {hash}.generation.json
    {hash}.extraction.json
    {hash}.score.json

The hash is a UUID4 assigned at generation time, not a content hash: repeats
are intentionally stochastic, so a content hash would be meaningless (or
collide) across repeats.
"""

import csv
import json
import threading
import uuid
from pathlib import Path

from config import DATA_DIR, condition_id

MANIFEST_FIELDS = ["repeat_index", "hash", "status"]

# One lock per condition folder: generation runs threaded, and manifest.csv is
# rewritten whole on every append.
_manifest_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _manifest_lock(cdir: Path) -> threading.Lock:
    with _locks_guard:
        return _manifest_locks.setdefault(str(cdir), threading.Lock())


def condition_dir(recipe: str, adjective: str, scale: str) -> Path:
    path = DATA_DIR / condition_id(recipe, adjective, scale)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_manifest(cdir: Path) -> dict[int, dict]:
    manifest_path = cdir / "manifest.csv"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, newline="") as f:
        return {int(row["repeat_index"]): row for row in csv.DictReader(f)}


def write_manifest(cdir: Path, rows: dict[int, dict]) -> None:
    with open(cdir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for repeat_index in sorted(rows):
            writer.writerow(rows[repeat_index])


def record_generation(cdir: Path, repeat_index: int, payload: dict) -> str:
    """Write {hash}.generation.json and register it in the manifest."""
    gen_hash = str(uuid.uuid4())
    write_json(cdir / f"{gen_hash}.generation.json", payload | {"hash": gen_hash})
    with _manifest_lock(cdir):
        rows = read_manifest(cdir)
        rows[repeat_index] = {
            "repeat_index": repeat_index,
            "hash": gen_hash,
            "status": "generated",
        }
        write_manifest(cdir, rows)
    return gen_hash


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def read_json(path: Path):
    return json.loads(path.read_text())


def iter_generations():
    """Yield (condition_dir, repeat_index, hash) for every cached generation."""
    for cdir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for repeat_index, row in sorted(read_manifest(cdir).items()):
            if (cdir / f"{row['hash']}.generation.json").exists():
                yield cdir, repeat_index, row["hash"]
