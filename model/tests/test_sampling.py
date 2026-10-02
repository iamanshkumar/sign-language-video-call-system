import pandas as pd

from asl.data.sampling import select_sample


def _pool():
    rows = []
    for c, n in enumerate([31, 35, 40, 50, 60, 90, 160]):
        rows += [{"sample_id": f"w{c}_{i}", "word": f"w{c}"} for i in range(n)]
    return pd.DataFrame(rows)


def test_sample_is_reproducible_balanced_and_without_replacement():
    a, ta = select_sample(_pool(), seed=42, per_class=2, target_total=18)
    b, _ = select_sample(_pool().sample(frac=1, random_state=3), seed=42, per_class=2, target_total=18)
    assert a["sample_id"].tolist() == b["sample_id"].tolist()
    assert a["sample_id"].is_unique and len(a) == 18
    assert ta["sampled"].min() == 2 and ta["sampled"].max() == 3
    # the 2 extra slots go to the smallest and the largest class
    assert ta.loc["w0", "selection"] == "extra_smallest_class"
    assert ta.loc["w6", "selection"] == "extra_largest_class"
    c, _ = select_sample(_pool(), seed=1, per_class=2, target_total=18)
    assert c["sample_id"].tolist() != a["sample_id"].tolist()
