import numpy as np
import pandas as pd
import pytest

from alphora_pipeline.data import clean_series, prepare


def test_duplicate_policy_is_explicit(cfg):
    raw = pd.DataFrame({"start_time": [0, 0], "v": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        clean_series(raw, {"column": "v"}, cfg, "x")
    cfg["cleaning"]["duplicates"] = "last"
    s, q = clean_series(raw, {"column": "v"}, cfg, "x")
    assert s.iloc[0] == 2 and q["duplicates"] == 1


def test_availability_lag_never_backfills_and_missing_prices_fail(cfg):
    cfg.update(start="2021-01-01T00:00Z", end="2021-01-01T04:00Z", warmup_bars=0)
    cfg["sources"]["input"]["availability_lag_minutes"] = 30
    cfg["cleaning"].update(missing="keep", max_missing_fraction=1)
    ix = pd.date_range(cfg["start"], periods=4, freq="1h")
    minute = pd.date_range("2021-01-01", periods=400, freq="1min", tz="UTC")
    raw = {
        "input": pd.DataFrame({"start_time": ix, "v": [10, 20, 30, 40]}),
        "__price": pd.DataFrame({"start_time": minute, "close": np.arange(400) + 100}),
    }
    frame, quality = prepare(cfg, raw)
    assert np.isnan(frame.input.iloc[0]) and frame.input.iloc[1] == 10
    assert frame.price.iloc[0] == 229 and quality["price"]["execution_offset_minutes"] == 129
    raw["__price"] = raw["__price"].drop(index=129)
    with pytest.raises(ValueError, match="execution prices"):
        prepare(cfg, raw)


def test_forward_fill_is_bounded_and_never_uses_future(cfg):
    cfg.update(start="2021-01-01T00:00Z", end="2021-01-01T04:00Z", warmup_bars=0)
    cfg["cleaning"].update(missing="ffill", ffill_limit_bars=1, max_missing_fraction=1)
    cfg["execution"].update(price_shift_minutes=0, price_sampling="bar_start", price_frequency="1h")
    ix = pd.date_range(cfg["start"], periods=4, freq="1h")
    raw = {
        "input": pd.DataFrame({"start_time": ix[[0, 3]], "v": [1.0, 9.0]}),
        "__price": pd.DataFrame({"start_time": ix, "close": [100] * 4}),
    }
    f, q = prepare(cfg, raw)
    assert f.input.iloc[1] == 1 and np.isnan(f.input.iloc[2]) and f.input.iloc[3] == 9
    assert q["input"]["filled_rows"] == 1
