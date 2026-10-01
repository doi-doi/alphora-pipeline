from copy import deepcopy

import pytest

from alphora_pipeline.config import combinations, validate


def test_configuration_rejects_mistyped_fields_and_reserved_source(cfg):
    cfg["execution"]["fee_percent"] = 0.035
    with pytest.raises(ValueError, match="Unknown execution"):
        validate(cfg)
    del cfg["execution"]["fee_percent"]
    cfg["sources"]["price"] = {"path": "input.csv", "column": "v"}
    with pytest.raises(ValueError, match="reserved"):
        validate(cfg)


def test_factor_cannot_depend_on_cached_thresholds(cfg):
    cfg["factor"] = {
        "steps": [{"name": "bad", "op": "multiply", "inputs": ["input", {"param": "threshold_1"}]}],
        "output": "bad",
    }
    with pytest.raises(ValueError, match="threshold parameters"):
        validate(cfg)


def test_factor_grid_parameters_are_validated_and_threshold_axes_are_inner(cfg):
    cfg["grid"]["factor_window"] = [12, 24]
    cfg["factor"] = {
        "steps": [{"name": "z", "op": "zscore", "input": "input", "window": {"param": "factor_window"}}],
        "output": "z",
    }
    validate(cfg)
    combos = list(combinations(cfg))
    assert combos[0]["factor_window"] == combos[1]["factor_window"]


def test_invalid_grid_and_nonfinite_parameters_fail(cfg):
    original = deepcopy(cfg)
    cfg["grid"]["threshold_1"] = [float("nan")]
    with pytest.raises(ValueError, match="finite"):
        validate(cfg)
    original["grid"]["window"] = list(range(2, 30000))
    with pytest.raises(ValueError, match="max_combinations"):
        validate(original)


def test_rolling_transform_requires_window(cfg):
    cfg["factor"] = {"steps": [{"name": "z", "op": "zscore", "input": "input"}], "output": "z"}
    with pytest.raises(ValueError, match="requires window"):
        validate(cfg)
