from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from alphora_pipeline.config import DEFAULTS


@pytest.fixture
def cfg():
    config = deepcopy(DEFAULTS)
    config.update(
        {
            "name": "test_strategy",
            "start": "2021-01-01T00:00:00Z",
            "end": "2023-01-01T00:00:00Z",
            "warmup_bars": 100,
            "sources": {"input": {"path": "input.csv", "column": "v"}},
            "price": {"path": "price.csv", "column": "close"},
            "factor": {"steps": [], "output": "input"},
            "grid": {
                "model": ["zscore"],
                "window": [12, 24],
                "threshold_1": [0.5, 1.0],
                "threshold_2": [-1.0, -0.5],
            },
        }
    )
    config["optimization"].update(min_valid_bars=12, min_trade_events=1)
    config["robustness"].update(bootstrap_samples=20, placebo_samples=10)
    return config


@pytest.fixture
def frame():
    rng = np.random.default_rng(3661)
    ix = pd.date_range("2020-12-01", "2023-01-01", freq="1h", inclusive="left", tz="UTC", name="start_time")
    return pd.DataFrame(
        {
            "input": rng.normal(size=len(ix)),
            "price": 20000 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ix)))),
            "execution_time": ix + pd.Timedelta(minutes=129),
        },
        index=ix,
    )
