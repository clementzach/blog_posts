import random

from datasets import load_dataset

SEED = 42


def load_mbpp_sample(n: int | None = 200) -> list[dict]:
    """Load `n` problems, or all of them if `n` is None."""
    ds = load_dataset("evalplus/mbppplus", split="test")
    if n is None:
        problems = [dict(row) for row in ds]
        print(f"Loaded {len(problems)} problems (all)")
        return problems
    rng = random.Random(SEED)
    population = list(range(len(ds)))
    k = min(n, len(population))
    indices = rng.sample(population, k)
    problems =  [dict(ds[i]) for i in sorted(indices)]
    print(f"Loaded {len(problems)} problems (seed={SEED})")
    return problems
