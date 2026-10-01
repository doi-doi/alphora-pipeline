import numpy as np
import pandas as pd

from alphora_pipeline.walk_forward import run_analysis, training_window


def test_training_excludes_prices_after_selection_boundary(cfg, frame):
    boundary = pd.Timestamp("2022-01-01", tz="UTC")
    w = training_window(frame, cfg, boundary)
    assert w["purged_bars"] == 2
    ix = frame.index[(frame.index >= w["train_start"]) & (frame.index < w["train_end"])]
    assert (frame.loc[ix, "execution_time"] < boundary).all()


def test_holdout_changes_cannot_change_recommendation_or_fold_parameters(cfg, frame):
    a = run_analysis(frame, cfg)
    cutoff = a["recommendation"]["selection_boundary"]
    bframe = frame.copy()
    future = bframe.execution_time >= cutoff
    bframe.loc[future, "price"] *= np.linspace(1, 10, int(future.sum()))
    bframe.loc[bframe.index >= cutoff, "input"] *= -1000
    b = run_analysis(bframe, cfg)
    assert a["recommendation"]["selected"] == b["recommendation"]["selected"]
    assert a["selected_parameters"] == b["selected_parameters"]
    assert a["holdout_metrics"]["total_return"] != b["holdout_metrics"]["total_return"]
    assert a["oos"].index.max() < a["holdout"].index.min()
    pd.testing.assert_frame_equal(a["oos"], b["oos"])
    assert (frame.loc[a["oos"].index, "execution_time"] < cutoff).all()


def test_warmup_guard_skips_a_fold_instead_of_manufacturing_training_edge(cfg, frame):
    frame = frame.copy()
    frame.loc[frame.index < pd.Timestamp("2021-07-01", tz="UTC"), "input"] = np.nan
    result = run_analysis(frame, cfg)
    assert result["folds"][0]["status"] == "skipped"
    assert result["coverage"] < 1
