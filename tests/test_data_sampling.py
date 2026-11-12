import numpy as np
import pandas as pd

from data.make_data import _reservoir_add


def test_reservoir_sampling_is_seeded():
    source = pd.DataFrame({"value": range(100)})
    first, first_scores = _reservoir_add(
        None, np.empty(0), source, 10, np.random.default_rng(42)
    )
    second, second_scores = _reservoir_add(
        None, np.empty(0), source, 10, np.random.default_rng(42)
    )
    assert first["value"].tolist() == second["value"].tolist()
    assert first_scores.tolist() == second_scores.tolist()
    assert len(first) == 10
