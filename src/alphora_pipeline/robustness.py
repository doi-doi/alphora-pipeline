"""Conditional OOS diagnostics; none of these scores reselect or reject parameters."""

from itertools import pairwise

import numpy as np
import pandas as pd

from .config import years_per_bar
from .engine import backtest
from .metrics import period_metrics, summarize
from .optimize import key, neighbors


def block_sample(values, block_size, rng):
    n = len(values)
    starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
    indices = (starts[:, None] + np.arange(block_size)) % n
    return values[indices.ravel()[:n]]


def run_robustness(frame, cfg, analysis):
    opts, exe = cfg["robustness"], cfg["execution"]
    annualizer = years_per_bar(cfg)
    factory = analysis["factory"]
    tested = [w for w in analysis["folds"] if w["status"] == "tested"]
    scenarios = []
    configs = [("cost", mult, exe["slippage_bps"], 0) for mult in opts["cost_multipliers"]]
    configs += [("slippage", 1.0, slip, 0) for slip in opts["slippage_bps"]]
    configs += [("execution_delay", 1.0, exe["slippage_bps"], int(delay)) for delay in opts["delay_bars"]]
    for kind, mult, slip, delay in configs:
        parts = []
        for w in tested:
            pos, _ = factory.positions(w["selected"]["parameters"])
            part = backtest(
                frame["price"], pos, w["test_start"], w["test_end"], exe["fee_bps"] * mult, slip, delay
            )
            parts.append(part)
        result = summarize(pd.concat(parts), annualizer)
        scenarios.append(
            {"scenario": kind, "fee_multiplier": mult, "slippage_bps": slip, "delay_bars": delay, **result}
        )
    # Parameter perturbations use only the final pre-holdout training window.
    w = analysis["recommendation"]
    center = w["selected"]["parameters"]
    perturbations = []
    for params in [center, *neighbors(center, cfg["grid"])]:
        if cfg["signal"]["overlap"] == "reject" and params["threshold_1"] <= params["threshold_2"]:
            continue
        pos, _ = factory.positions(params)
        part = backtest(
            frame["price"], pos, w["train_start"], w["train_end"], exe["fee_bps"], exe["slippage_bps"]
        )
        perturbations.append(
            {"parameters": params, "is_selected": params == center, **summarize(part, annualizer)}
        )
    rng = np.random.default_rng(opts["seed"])
    oos = analysis["oos"]
    p = oos["pnl"].to_numpy()
    block = min(opts["block_bars"], len(p))
    samples = []
    for _ in range(opts["bootstrap_samples"]):
        sample = block_sample(p, block, rng)
        std = sample.std(ddof=1)
        equity = sample.cumsum()
        dd = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
        samples.append(
            [sample.mean() / std * np.sqrt(annualizer) if std > 0 else np.nan, sample.sum(), dd.min()]
        )
    matrix = np.asarray(samples)
    intervals = {
        k: {
            "p025": float(np.nanquantile(matrix[:, i], 0.025)),
            "median": float(np.nanmedian(matrix[:, i])),
            "p975": float(np.nanquantile(matrix[:, i], 0.975)),
        }
        for i, k in enumerate(["sharpe", "total_return", "max_drawdown"])
    }
    # Fixed selected decisions versus circularly shifted asset returns, within each tested fold.
    # These are conditional placebo diagnostics, not a re-optimization-adjusted significance test.
    null_sharpes = []
    fold_parts = [oos.loc[oos["fold"] == w["fold"]] for w in tested]
    for _ in range(opts["placebo_samples"]):
        chunks = []
        for part in fold_parts:
            shift = int(rng.integers(1, len(part)))
            returns = np.roll(part["asset_return"].to_numpy(), shift)
            chunks.append(part["held_position"].to_numpy() * returns - part["cost"].to_numpy())
        null = np.concatenate(chunks)
        std = null.std(ddof=1)
        if std > 0:
            null_sharpes.append(float(null.mean() / std * np.sqrt(annualizer)))
    observed = analysis["oos_metrics"]["sharpe"]
    placebo = {
        "samples": len(null_sharpes),
        "observed_sharpe": observed,
        "fraction_at_or_above_observed": float(
            (sum(x >= observed for x in null_sharpes) + 1) / (len(null_sharpes) + 1)
        )
        if null_sharpes and observed is not None
        else None,
        "null_sharpe_p95": float(np.quantile(null_sharpes, 0.95)) if null_sharpes else None,
        "interpretation": "Conditional fixed-decision placebo; does not correct for grid/model selection, revised source data or repeated research.",
    }
    selected = [w["selected"]["parameters"] for w in tested]
    return {
        "scenarios": scenarios,
        "parameter_neighbors": perturbations,
        "yearly_oos": period_metrics(oos, annualizer),
        "bootstrap": {
            "samples": opts["bootstrap_samples"],
            "block_bars": block,
            "intervals": intervals,
            "interpretation": "Moving circular block bootstrap of stitched OOS PnL, conditional on the completed selection. Not selection-adjusted or a forecast.",
        },
        "placebo": placebo,
        "parameter_stability": {
            "unique_selected_combinations": len({key(p) for p in selected}),
            "tested_folds": len(tested),
            "parameter_changes": sum(a != b for a, b in pairwise(selected)),
        },
    }
