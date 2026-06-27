import random

from datasets import load_dataset

SEED = 42


def load_mbpp_sample(n: int = 200) -> list[dict]:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rng = random.Random(SEED)
    population = list(range(len(ds)))
    k = min(n, len(population))
    indices = rng.sample(population, k)
    problems =  [dict(ds[i]) for i in sorted(indices)]
    print(f"Loaded {len(problems)} problems (seed={SEED})")
    return problems
