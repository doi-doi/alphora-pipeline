from alphora_pipeline.optimize import select


def record(window, score):
    return {
        "parameters": {"model": "zscore", "window": window, "threshold_1": 1.0, "threshold_2": -1.0},
        "eligible": True,
        "sharpe": score,
    }


def test_connected_plateau_prefers_broad_region_over_isolated_peak(cfg):
    cfg["grid"] = {
        "model": ["zscore"],
        "window": [10, 20, 30, 40, 50],
        "threshold_1": [1.0],
        "threshold_2": [-1.0],
    }
    result = select(
        [record(10, 2.0), record(20, 0.0), record(30, 1.91), record(40, 1.9), record(50, 1.89)], cfg
    )
    assert result["parameters"]["window"] == 40 and result["plateau_size"] == 3


def test_empty_eligible_training_does_not_pick_grid_center(cfg):
    r = record(12, None)
    r["eligible"] = False
    assert select([r], cfg) is None


def test_negative_best_sharpe_tolerance_is_absolute(cfg):
    cfg["grid"] = {"model": ["zscore"], "window": [10, 20], "threshold_1": [1.0], "threshold_2": [-1.0]}
    result = select([record(10, -1.0), record(20, -1.05)], cfg)
    assert result["plateau_size"] == 2
