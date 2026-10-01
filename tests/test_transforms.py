import numpy as np
import pandas as pd
import pytest

from alphora_pipeline.transforms import SignalFactory, factor, model


@pytest.mark.parametrize(
    "kind",
    [
        "identity",
        "zscore",
        "ezscorev1",
        "percentile_rank",
        "percentilerank_meannorm",
        "minmax",
        "robust_zscore",
    ],
)
def test_model_is_prefix_causal(kind):
    values = pd.Series(np.random.default_rng(4).normal(size=100))
    original = model(values, kind, 10)
    changed = values.copy()
    changed.iloc[60:] = 1000
    pd.testing.assert_series_equal(original.iloc[:60], model(changed, kind, 10).iloc[:60])


def test_graph_log_diff_zscore_and_mask_matches_explicit_formula():
    values = pd.DataFrame({"skew": np.sin(np.arange(100) / 10), "oi": np.exp(np.arange(100) / 100)})
    graph = {
        "steps": [
            {"name": "skew_d", "op": "diff", "input": "skew", "periods": 2},
            {"name": "z", "op": "zscore", "input": "skew_d", "window": 10},
            {"name": "negative", "op": "negate", "input": "z"},
            {"name": "oi_log", "op": "log", "input": "oi"},
            {"name": "oi_d", "op": "diff", "input": "oi_log", "periods": 2},
            {"name": "gate", "op": "gt", "inputs": ["oi_d", 0]},
            {"name": "result", "op": "multiply", "inputs": ["negative", "gate"]},
        ],
        "output": "result",
    }
    delta = values["skew"].diff(2)
    expected = -(delta - delta.rolling(10).mean()) / delta.rolling(10).std() * (np.log(values.oi).diff(2) > 0)
    np.testing.assert_allclose(factor(values, graph, {}), expected, equal_nan=True)


def test_clip_uses_only_past_quantiles():
    x = pd.DataFrame({"x": [1.0, 2.0, 3.0, 1000.0, 4.0]})
    graph = {
        "steps": [{"name": "clip", "op": "rolling_clip", "input": "x", "window": 3, "lower": 0, "upper": 1}],
        "output": "clip",
    }
    assert factor(x, graph, {})[3] == 3.0


def test_overlap_short_priority_and_missing_signal_flat(cfg):
    frame = pd.DataFrame({"input": [-1.0, 0.0, 1.0, np.nan]})
    p = {"model": "identity", "window": 12, "threshold_1": -0.5, "threshold_2": 0.5}
    positions, _ = SignalFactory(frame, cfg).positions(p)
    assert positions.tolist() == [-1.0, -1.0, 1.0, 0.0]
