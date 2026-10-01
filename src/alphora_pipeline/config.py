"""Strict JSON configuration; all relative paths resolve against the config file."""

import json
import math
import re
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd

MODELS = {
    "identity",
    "zscore",
    "ezscorev1",
    "percentile_rank",
    "percentilerank_meannorm",
    "minmax",
    "robust_zscore",
}
OPS = {
    "identity",
    "log",
    "log1p",
    "diff",
    "pct_change",
    "abs",
    "negate",
    "square",
    "sqrt",
    "rolling_mean",
    "rolling_std",
    "zscore",
    "percentile_rank",
    "rolling_clip",
    "add",
    "subtract",
    "multiply",
    "divide",
    "gt",
    "lt",
    "and",
}
DEFAULTS = {
    "frequency": "1h",
    "warmup_bars": 5000,
    "cleaning": {
        "duplicates": "error",
        "missing": "error",
        "ffill_limit_bars": 1,
        "max_missing_fraction": 0.0,
    },
    "execution": {
        "fee_bps": 3.5,
        "slippage_bps": 0.0,
        "price_shift_minutes": 70,
        "price_sampling": "bar_last",
        "price_frequency": "1min",
        "invalid_signal": "flat",
    },
    "signal": {"logic": "trend", "side": "both", "overlap": "short_priority"},
    "optimization": {
        "absolute_tolerance": 0.1,
        "relative_tolerance": 0.1,
        "min_valid_bars": 168,
        "min_trade_events": 5,
        "max_combinations": 20000,
        "selection": "connected_plateau",
    },
    "walk_forward": {
        "train_months": 6,
        "test_months": 3,
        "embargo_bars": 0,
        "holdout_months": 3,
        "min_test_bars": 168,
    },
    "robustness": {
        "bootstrap_samples": 500,
        "block_bars": 168,
        "seed": 3661,
        "cost_multipliers": [1.0, 1.5, 2.0],
        "slippage_bps": [0, 2, 5],
        "delay_bars": [0, 1, 2],
        "placebo_samples": 100,
    },
}


def timestamp(value):
    t = pd.Timestamp(value)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    validate(cfg)
    return cfg, path.parent


def validate(cfg):
    required = {"name", "start", "end", "sources", "price", "factor", "grid"}
    if required - set(cfg):
        raise ValueError(f"Missing configuration fields: {sorted(required - set(cfg))}")

    def check_finite(value):
        if isinstance(value, dict):
            for v in value.values():
                check_finite(v)
        elif isinstance(value, list):
            for v in value:
                check_finite(v)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Configuration numbers must be finite")

    check_finite(cfg)
    for section, options in DEFAULTS.items():
        if isinstance(options, dict) and set(cfg[section]) - set(options):
            raise ValueError(f"Unknown {section} fields: {sorted(set(cfg[section]) - set(options))}")
    allowed = set(DEFAULTS) | {
        "name",
        "start",
        "end",
        "sources",
        "price",
        "factor",
        "grid",
        "notes",
        "baseline_parameters",
    }
    if set(cfg) - allowed:
        raise ValueError(f"Unknown configuration fields: {sorted(set(cfg) - allowed)}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg.get("name", "")):
        raise ValueError("name must contain only letters, digits, underscores or hyphens")
    if timestamp(cfg["end"]) <= timestamp(cfg["start"]):
        raise ValueError("end must be after start (end is exclusive)")
    freq = pd.Timedelta(cfg["frequency"])
    if (
        freq <= pd.Timedelta(0)
        or timestamp(cfg["start"]).value % freq.value
        or timestamp(cfg["end"]).value % freq.value
    ):
        raise ValueError("frequency must be positive and start aligned to its UTC grid")
    if cfg["warmup_bars"] < 0 or not isinstance(cfg["warmup_bars"], int):
        raise ValueError("warmup_bars must be a nonnegative integer")
    if not cfg["sources"] or not isinstance(cfg["sources"], dict):
        raise ValueError("At least one named endpoint or file source is required")
    if set(cfg["sources"]) & {"price", "execution_time", "start_time"}:
        raise ValueError("Source alias conflicts with a reserved accounting column")
    for name, spec in {**cfg["sources"], "price": cfg["price"]}.items():
        allowed_source = {
            "topic",
            "path",
            "column",
            "timestamp_column",
            "timestamp_unit",
            "availability_lag_minutes",
        }
        if set(spec) - allowed_source:
            raise ValueError(f"{name}: unknown source fields: {sorted(set(spec) - allowed_source)}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid source name: {name}")
        if bool(spec.get("topic")) == bool(spec.get("path")):
            raise ValueError(f"{name}: set exactly one of topic or path")
        if not spec.get("column"):
            raise ValueError(f"{name}: column is required")
        if float(spec.get("availability_lag_minutes", 0)) < 0:
            raise ValueError("Source availability lag cannot be negative")
        if spec.get("timestamp_unit", "ms") not in {"s", "ms", "us", "ns", "iso"}:
            raise ValueError("timestamp_unit must be s, ms, us, ns or iso")
    cleaning = cfg["cleaning"]
    if cleaning["duplicates"] not in {"error", "first", "last"} or cleaning["missing"] not in {
        "error",
        "ffill",
        "keep",
    }:
        raise ValueError("Unknown cleaning policy")
    if not 0 <= cleaning["max_missing_fraction"] <= 1 or cleaning["ffill_limit_bars"] < 1:
        raise ValueError("Invalid missing-data tolerance or forward-fill limit")
    exe = cfg["execution"]
    if any(exe[k] < 0 for k in ("fee_bps", "slippage_bps", "price_shift_minutes")):
        raise ValueError("Costs and price shift must be nonnegative")
    if exe["price_sampling"] not in {"bar_last", "bar_start"} or exe["invalid_signal"] not in {
        "flat",
        "hold",
    }:
        raise ValueError("Invalid execution policy")
    pf = pd.Timedelta(exe["price_frequency"])
    if (
        pf <= pd.Timedelta(0)
        or freq.value % pf.value
        or pd.Timedelta(minutes=exe["price_shift_minutes"]).value % pf.value
    ):
        raise ValueError("Signal interval and shift must be integer multiples of price_frequency")
    if cfg["signal"]["logic"] not in {"trend", "trend_reverse", "fast", "fast_reverse"} or cfg["signal"][
        "side"
    ] not in {"both", "long", "short"}:
        raise ValueError("Unsupported position logic or side")
    if cfg["signal"]["overlap"] not in {"short_priority", "reject"}:
        raise ValueError("overlap must be short_priority or reject")
    wf = cfg["walk_forward"]
    if any(not isinstance(wf[k], int) or wf[k] < 1 for k in ("train_months", "test_months", "min_test_bars")):
        raise ValueError("Walk-forward months and minimum test bars must be positive integers")
    if any(not isinstance(wf[k], int) or wf[k] < 0 for k in ("holdout_months", "embargo_bars")):
        raise ValueError("Holdout and embargo cannot be negative")
    opt = cfg["optimization"]
    if opt["selection"] not in {"connected_plateau", "legacy_grid_center"}:
        raise ValueError("Unknown plateau selection method")
    if (
        any(opt[k] < 0 for k in ("absolute_tolerance", "relative_tolerance", "min_trade_events"))
        or opt["min_valid_bars"] < 2
    ):
        raise ValueError("Invalid training eligibility or plateau tolerance")
    grid = cfg["grid"]
    if not {"model", "window", "threshold_1", "threshold_2"}.issubset(grid):
        raise ValueError("grid needs model, window, threshold_1 and threshold_2")
    for key, vals in grid.items():
        if not isinstance(vals, list) or not vals or len({json.dumps(v) for v in vals}) != len(vals):
            raise ValueError(f"{key}: grid axis must be a nonempty list of unique values")
        if key != "model" and not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            raise ValueError(f"{key}: grid axis must be numeric")
    if any(m not in MODELS for m in grid["model"]) or any(
        not isinstance(w, int) or w < 2 for w in grid["window"]
    ):
        raise ValueError("Unknown model or invalid window")
    if math.prod(len(v) for v in grid.values()) > opt["max_combinations"]:
        raise ValueError("Parameter grid exceeds max_combinations")
    if not any(combinations(cfg)):
        raise ValueError("No valid threshold combinations remain")
    nodes = cfg["factor"].get("steps", [])
    if set(cfg["factor"]) - {"steps", "output"}:
        raise ValueError("Unknown factor fields")
    known = set(cfg["sources"])
    for node in nodes:
        if node["name"] in known or node["op"] not in OPS:
            raise ValueError(f"Duplicate node or unknown transform: {node}")
        inputs = node.get("inputs", [node.get("input")])
        binary = node["op"] in {"add", "subtract", "multiply", "divide", "gt", "lt", "and"}
        if len(inputs) != (2 if binary else 1) or not isinstance(inputs[0], str):
            raise ValueError("Transform needs a named first input and the correct number of inputs")
        if (
            node["op"] in {"rolling_mean", "rolling_std", "zscore", "percentile_rank", "rolling_clip"}
            and "window" not in node
        ):
            raise ValueError("Rolling transform requires window")
        if node["op"] == "rolling_clip" and not 0 <= node.get("lower", 0.01) < node.get("upper", 0.99) <= 1:
            raise ValueError("Rolling clipping needs 0 <= lower < upper <= 1")
        for term in inputs:
            if isinstance(term, str) and term not in known:
                raise ValueError(f"Unknown or forward-referenced transform input: {term}")
            if isinstance(term, dict) and (set(term) != {"param"} or term["param"] not in grid):
                raise ValueError(f"Unknown factor parameter: {term}")
            if isinstance(term, dict) and term["param"] in {"model", "threshold_1", "threshold_2"}:
                raise ValueError(
                    "Factor graph cannot depend on model/threshold parameters; use a dedicated factor grid axis"
                )
        for key in ("window", "periods"):
            v = node.get(key)
            if isinstance(v, dict) and (set(v) != {"param"} or v["param"] not in grid):
                raise ValueError(f"Unknown factor parameter: {v}")
            if isinstance(v, dict) and v["param"] in {"model", "threshold_1", "threshold_2"}:
                raise ValueError("Factor windows cannot depend on model/threshold parameters")
            options = grid[v["param"]] if isinstance(v, dict) else [v] if v is not None else []
            if any(not isinstance(n, int) or n < (2 if key == "window" else 1) for n in options):
                raise ValueError("Rolling windows must be >=2 and periods positive integers")
        known.add(node["name"])
    if cfg["factor"]["output"] not in known:
        raise ValueError("factor.output must reference a source or transform node")
    rob = cfg["robustness"]
    if any(rob[k] < 1 for k in ("bootstrap_samples", "block_bars")) or rob["placebo_samples"] < 0:
        raise ValueError("Invalid resampling settings")
    if any(v < 0 for k in ("cost_multipliers", "slippage_bps", "delay_bars") for v in rob[k]):
        raise ValueError("Robustness costs/delays cannot be negative")
    if any(not isinstance(v, int) for v in rob["delay_bars"]):
        raise ValueError("Robustness delays must be integer bars")
    if cfg.get("baseline_parameters"):
        base = cfg["baseline_parameters"]
        if (
            not {"model", "window", "threshold_1", "threshold_2"}.issubset(base)
            or base["model"] not in MODELS
            or not isinstance(base["window"], int)
            or base["window"] < 2
        ):
            raise ValueError("baseline_parameters needs a supported model, window and thresholds")


def combinations(cfg):
    # Keep threshold axes innermost to reuse each expensive factor/model history.
    keys = [k for k in cfg["grid"] if k not in {"threshold_1", "threshold_2"}] + [
        "threshold_1",
        "threshold_2",
    ]
    for values in product(*(cfg["grid"][k] for k in keys)):
        p = dict(zip(keys, values, strict=True))
        if cfg["signal"]["overlap"] == "reject" and p["threshold_1"] <= p["threshold_2"]:
            continue
        yield p


def years_per_bar(cfg):
    return 365.0 * 24 * 3600 / pd.Timedelta(cfg["frequency"]).total_seconds()
