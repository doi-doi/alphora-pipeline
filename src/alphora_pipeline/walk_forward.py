"""Untouched fold tests, embargo and execution-time purging, then a final holdout."""

import logging

import pandas as pd

from .config import combinations, timestamp, years_per_bar
from .engine import backtest
from .metrics import summarize, with_equity
from .optimize import eligibility, key, select
from .transforms import SignalFactory

LOG = logging.getLogger(__name__)


def training_window(frame, cfg, boundary):
    wf = cfg["walk_forward"]
    start = boundary - pd.DateOffset(months=wf["train_months"])
    end = boundary - wf["embargo_bars"] * pd.Timedelta(cfg["frequency"])
    available = frame.index[(frame.index >= start) & (frame.index < end) & (frame["execution_time"] < end)]
    # Empty training windows are recorded as skipped, not awarded a default parameter.
    effective_end = available[-1] + pd.Timedelta(cfg["frequency"]) if len(available) else start
    return {
        "train_start": start,
        "train_end": effective_end,
        "selection_boundary": boundary,
        "purged_bars": int(
            ((frame.index >= start) & (frame.index < end) & (frame["execution_time"] >= end)).sum()
        ),
    }


def schedule(frame, cfg):
    wf = cfg["walk_forward"]
    start, end = timestamp(cfg["start"]), timestamp(cfg["end"])
    holdout_start = end - pd.DateOffset(months=wf["holdout_months"]) if wf["holdout_months"] else end
    # Round the holdout boundary up onto the signal grid.
    holdout_start = holdout_start.ceil(pd.Timedelta(cfg["frequency"]))
    first_test = start + pd.DateOffset(months=wf["train_months"])
    folds = []
    cursor = first_test
    while cursor < holdout_start:
        test_end = min(cursor + pd.DateOffset(months=wf["test_months"]), holdout_start)
        scheduled_end = test_end
        # The final development test must not consume holdout execution prices either.
        if test_end == holdout_start and wf["holdout_months"]:
            available = frame.index[
                (frame.index >= cursor) & (frame.index < test_end) & (frame["execution_time"] < holdout_start)
            ]
            test_end = available[-1] + pd.Timedelta(cfg["frequency"]) if len(available) else cursor
        if ((frame.index >= cursor) & (frame.index < test_end)).sum() >= wf["min_test_bars"]:
            folds.append(
                {
                    "fold": len(folds) + 1,
                    **training_window(frame, cfg, cursor),
                    "test_start": cursor,
                    "test_end": test_end,
                }
            )
        cursor = scheduled_end
    if not folds:
        raise ValueError(
            "Not enough dates for walk-forward development plus holdout; extend dates or shorten windows"
        )
    recommendation = {"fold": "recommendation", **training_window(frame, cfg, holdout_start)}
    return folds, recommendation, holdout_start


def window(full, start, end, cost_bps):
    """Reset a precomputed causal history to the same flat-start accounting as engine.backtest."""
    p = full.loc[(full.index >= start) & (full.index < end)].copy()
    if not p.empty:
        first = p.index[0]
        turn = abs(p.loc[first, "position"])
        cost = turn * cost_bps / 10000
        p.loc[first, ["held_position", "asset_return", "gross_pnl"]] = 0
        p.loc[first, "turnover"] = turn
        p.loc[first, "cost"] = cost
        p.loc[first, "pnl"] = -cost
    return p


def run_analysis(frame, cfg):
    annualizer = years_per_bar(cfg)
    exe, opt = cfg["execution"], cfg["optimization"]
    factory = SignalFactory(frame, cfg)
    folds, recommendation, holdout_start = schedule(frame, cfg)
    windows = [*folds, recommendation]
    scores = [[] for _ in windows]
    start, end = timestamp(cfg["start"]), timestamp(cfg["end"])
    cost_bps = exe["fee_bps"] + exe["slippage_bps"]
    total = sum(1 for _ in combinations(cfg))
    for n, parameters in enumerate(combinations(cfg), start=1):
        positions, valid = factory.positions(parameters)
        full = backtest(frame["price"], positions, start, end, exe["fee_bps"], exe["slippage_bps"])
        for i, w in enumerate(windows):
            part = window(full, w["train_start"], w["train_end"], cost_bps)
            mask = (frame.index >= w["train_start"]) & (frame.index < w["train_end"])
            valid_bars = int(valid[mask].sum())
            if part.empty:
                records = {
                    "sharpe": None,
                    "trade_events": 0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "rows": 0,
                }
                good, reasons = False, ["empty_training_window"]
            else:
                records = summarize(part, annualizer)
                good, reasons = eligibility(records, valid_bars, opt)
            scores[i].append(
                {
                    "parameters": parameters,
                    "eligible": good,
                    "reasons": reasons,
                    "valid_bars": valid_bars,
                    **records,
                }
            )
        if n == 1 or n % 100 == 0 or n == total:
            LOG.info("Training grid %d/%d candidates scored", n, total)
    stitched, fold_results = [], []
    selected_parameters = {}
    for w, records in zip(folds, scores[:-1], strict=True):
        chosen = select(records, cfg)
        if chosen is None:
            fold_results.append(
                {
                    **w,
                    "status": "skipped",
                    "reason": "No eligible training candidate; warm-up, trades or Sharpe guard failed",
                }
            )
            continue
        p = chosen["parameters"]
        positions, valid = factory.positions(p)
        test = backtest(
            frame["price"], positions, w["test_start"], w["test_end"], exe["fee_bps"], exe["slippage_bps"]
        )
        test["fold"] = w["fold"]
        test["parameters"] = key(p)
        stitched.append(test)
        selected_parameters[w["fold"]] = p
        test_valid = int(valid[(frame.index >= w["test_start"]) & (frame.index < w["test_end"])].sum())
        fold_results.append(
            {
                **w,
                "status": "tested",
                "selected": chosen,
                "test_valid_bars": test_valid,
                "test_metrics": summarize(test, annualizer),
            }
        )
    if not stitched:
        raise ValueError(
            "Every fold failed training eligibility; add warm-up/history, revise grid or eligibility settings"
        )
    oos = with_equity(pd.concat(stitched).sort_index())
    if oos.index.has_duplicates:
        raise AssertionError("Overlapping test windows")
    final = select(scores[-1], cfg)
    if final is None:
        raise ValueError("No eligible candidate for the final pre-holdout training window")
    parameters = final["parameters"]
    positions, _ = factory.positions(parameters)
    # Full-history curve is descriptive hindsight, never labelled out of sample.
    descriptive = with_equity(
        backtest(frame["price"], positions, start, end, exe["fee_bps"], exe["slippage_bps"])
    )
    holdout = None
    if holdout_start < end:
        holdout = with_equity(
            backtest(frame["price"], positions, holdout_start, end, exe["fee_bps"], exe["slippage_bps"])
        )
    expected = sum(
        int(((frame.index >= w["test_start"]) & (frame.index < w["test_end"])).sum()) for w in folds
    )
    baseline = None
    if cfg.get("baseline_parameters"):
        p = {**parameters, **cfg["baseline_parameters"]}
        pos, _ = factory.positions(p)
        baseline = with_equity(backtest(frame["price"], pos, start, end, exe["fee_bps"], exe["slippage_bps"]))
    return {
        "folds": fold_results,
        "training_scores": scores,
        "training_windows": windows,
        "oos": oos,
        "oos_metrics": summarize(oos, annualizer),
        "coverage": len(oos) / expected,
        "recommendation": {**recommendation, "selected": final, "parameters_as_of": holdout_start},
        "holdout": holdout,
        "holdout_metrics": summarize(holdout, annualizer) if holdout is not None else None,
        "descriptive": descriptive,
        "descriptive_metrics": summarize(descriptive, annualizer),
        "baseline": baseline,
        "baseline_metrics": summarize(baseline, annualizer) if baseline is not None else None,
        "factory": factory,
        "selected_parameters": selected_parameters,
    }
