"""Deterministic synthetic offline example; never presented as market evidence."""

from pathlib import Path

import numpy as np
import pandas as pd

from .reporting import save_json


def create_demo(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3661)
    ix = pd.date_range("2020-11-01", "2023-01-01", freq="1h", inclusive="left", tz="UTC")
    factor = np.zeros(len(ix))
    for i in range(1, len(ix)):
        factor[i] = 0.85 * factor[i - 1] + rng.normal(0, 0.5)
    returns = 0.00015 * np.sign(np.r_[0, factor[:-1]]) + rng.normal(0, 0.002, len(ix))
    price = 20000 * np.exp(np.cumsum(returns))
    pd.DataFrame({"start_time": ix, "value": factor}).to_csv(directory / "factor.csv", index=False)
    pd.DataFrame({"start_time": ix, "close": price}).to_csv(directory / "price.csv", index=False)
    cfg = {
        "name": "synthetic_demo",
        "start": "2021-01-01T00:00:00Z",
        "end": "2023-01-01T00:00:00Z",
        "warmup_bars": 500,
        "sources": {"endpoint": {"path": "factor.csv", "column": "value", "timestamp_unit": "iso"}},
        "price": {"path": "price.csv", "column": "close", "timestamp_unit": "iso"},
        "factor": {"steps": [], "output": "endpoint"},
        "grid": {
            "model": ["zscore"],
            "window": [24, 48, 72],
            "threshold_1": [0.5, 1.0, 1.5],
            "threshold_2": [-0.5, -1.0, -1.5],
        },
        "execution": {"price_shift_minutes": 0, "price_sampling": "bar_start", "price_frequency": "1h"},
        "notes": [
            "SYNTHETIC DEMONSTRATION: generated data with an artificial relationship; these results are not market backtest evidence."
        ],
        "robustness": {"bootstrap_samples": 100, "placebo_samples": 50},
    }
    path = directory / "strategy.json"
    save_json(path, cfg)
    return path
