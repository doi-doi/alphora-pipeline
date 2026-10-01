"""Causal named transformation graph. No eval, backfill or global fitting."""

import numpy as np
import pandas as pd


def rolling_zscore(x, window):
    r = x.rolling(window, min_periods=window)
    return (x - r.mean()) / r.std(ddof=1).replace(0, np.nan)


def rank(x, window):
    return x.rolling(window, min_periods=window).rank(pct=True, method="average")


def resolve(value, parameters, columns):
    if isinstance(value, dict):
        return parameters[value["param"]]
    if isinstance(value, str):
        return columns[value]
    return value


def factor(frame, specification, parameters):
    columns = dict(frame.items())
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for step in specification.get("steps", []):
            args = [resolve(a, parameters, columns) for a in step.get("inputs", [step.get("input")])]
            x = args[0]
            window = resolve(step.get("window"), parameters, columns)
            periods = resolve(step.get("periods", 1), parameters, columns)
            op = step["op"]
            if op == "identity":
                y = x
            elif op == "log":
                y = np.log(x.where(x > 0))
            elif op == "log1p":
                y = np.log1p(x.where(x > -1))
            elif op == "diff":
                y = x.diff(periods)
            elif op == "pct_change":
                y = x.pct_change(periods, fill_method=None)
            elif op == "abs":
                y = x.abs()
            elif op == "negate":
                y = -x
            elif op == "square":
                y = x * x
            elif op == "sqrt":
                y = np.sqrt(x.where(x >= 0))
            elif op == "rolling_mean":
                y = x.rolling(window, min_periods=window).mean()
            elif op == "rolling_std":
                y = x.rolling(window, min_periods=window).std(ddof=1)
            elif op == "zscore":
                y = rolling_zscore(x, window)
            elif op == "percentile_rank":
                y = rank(x, window)
            elif op == "rolling_clip":
                # Bounds estimated on strictly preceding rows, never a full-sample quantile.
                r = x.shift(1).rolling(window, min_periods=window)
                y = x.clip(r.quantile(step.get("lower", 0.01)), r.quantile(step.get("upper", 0.99)))
            elif op == "add":
                y = x + args[1]
            elif op == "subtract":
                y = x - args[1]
            elif op == "multiply":
                y = x * args[1]
            elif op == "divide":
                y = x / args[1]
            elif op == "gt":
                y = (x > args[1]).astype(float).where(pd.notna(x) & pd.notna(args[1]))
            elif op == "lt":
                y = (x < args[1]).astype(float).where(pd.notna(x) & pd.notna(args[1]))
            elif op == "and":
                y = ((x != 0) & (args[1] != 0)).astype(float).where(pd.notna(x) & pd.notna(args[1]))
            else:
                raise ValueError(f"Unknown transform {op}")
            if not isinstance(y, pd.Series):
                y = pd.Series(y, index=frame.index)
            columns[step["name"]] = y.replace([np.inf, -np.inf], np.nan)
    return columns[specification["output"]].astype(float)


def model(values, kind, window):
    if kind == "identity":
        y = values
    elif kind == "zscore":
        y = rolling_zscore(values, window)
    elif kind == "ezscorev1":
        e = values.ewm(span=window, adjust=False, min_periods=window)
        y = (values - e.mean()) / e.std(bias=False).replace(0, np.nan)
    elif kind == "percentile_rank":
        y = rank(values, window)
    elif kind == "percentilerank_meannorm":
        p = rank(values, window)
        r = p.rolling(window, min_periods=window)
        y = (p - r.mean()) / (r.max() - r.min()).replace(0, np.nan)
    elif kind == "minmax":
        r = values.rolling(window, min_periods=window)
        y = (values - r.min()) / (r.max() - r.min()).replace(0, np.nan)
    elif kind == "robust_zscore":
        r = values.rolling(window, min_periods=window)
        y = (values - r.median()) / ((r.quantile(0.75) - r.quantile(0.25)) / 1.349).replace(0, np.nan)
    else:
        raise ValueError(f"Unknown model {kind}")
    return y.replace([np.inf, -np.inf], np.nan)


class SignalFactory:
    """Only the current factor/model signature is cached; grid memory stays bounded."""

    def __init__(self, frame, cfg):
        self.frame, self.cfg = frame, cfg
        self.key, self.values = None, None

    def scores(self, parameters):
        key = tuple(sorted((k, v) for k, v in parameters.items() if k not in {"threshold_1", "threshold_2"}))
        if key != self.key:
            f = factor(self.frame, self.cfg["factor"], parameters)
            self.values = model(f, parameters["model"], parameters["window"])
            self.key = key
        return self.values

    def positions(self, parameters):
        values = self.scores(parameters)
        v = values.to_numpy()
        valid = np.isfinite(v)
        logic, side = self.cfg["signal"]["logic"], self.cfg["signal"]["side"]
        long = valid & (v >= parameters["threshold_1"])
        short = valid & (v <= parameters["threshold_2"])
        events = np.full(len(v), 0.0 if logic.startswith("fast") else np.nan)
        events[long] = 1
        events[short] = -1  # explicit short priority for crossed thresholds
        if logic.endswith("reverse"):
            events *= -1
        if side == "long":
            events[events == -1] = 0
        if side == "short":
            events[events == 1] = 0
        if self.cfg["execution"]["invalid_signal"] == "flat":
            events[~valid] = 0
        position = pd.Series(events, index=values.index).ffill().fillna(0)
        return position, valid
