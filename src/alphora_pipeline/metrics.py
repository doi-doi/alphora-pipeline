"""Additive metrics consistent with the previous backtests; ratios are nullable."""

import numpy as np


def ratio(a, b):
    return float(a / b) if b > 0 and np.isfinite(a / b) else None


def summarize(performance, annualizer):
    if performance.empty:
        raise ValueError("Cannot calculate metrics on an empty backtest")
    p = performance["pnl"].to_numpy(dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite PnL; missing returns must not be silently zeroed")
    equity = np.cumsum(p)
    dd = equity - np.maximum.accumulate(np.r_[0.0, equity])[1:]
    std = p.std(ddof=1) if len(p) > 1 else 0
    downside = np.sqrt(np.mean(np.minimum(p, 0) ** 2))
    ar = p.mean() * annualizer
    positive, negative = p[p > 0].sum(), -p[p < 0].sum()
    neg = dd < 0
    longest = current = 0
    for underwater in neg:
        current = current + 1 if underwater else 0
        longest = max(longest, current)
    turn = performance["turnover"].to_numpy()
    held = performance["held_position"].to_numpy()
    return {
        "rows": len(p),
        "start": performance.index[0].isoformat(),
        "end_inclusive": performance.index[-1].isoformat(),
        "sharpe": ratio(p.mean() * np.sqrt(annualizer), std),
        "sortino": ratio(p.mean() * np.sqrt(annualizer), downside),
        "total_return": float(equity[-1]),
        "annual_return": float(ar),
        "max_drawdown": float(dd.min()),
        "calmar": ratio(ar, abs(dd.min())),
        "annual_volatility": float(std * np.sqrt(annualizer)),
        "bar_win_rate": float(np.mean(p[p != 0] > 0)) if np.any(p != 0) else None,
        "bar_profit_factor": ratio(positive, negative),
        "trade_events": int(np.count_nonzero(turn)),
        "turnover_units": float(turn.sum()),
        "fees_and_slippage": float(performance["cost"].sum()),
        "exposure": float(np.mean(held != 0)),
        "long_exposure": float(np.mean(held > 0)),
        "short_exposure": float(np.mean(held < 0)),
        "longest_drawdown_bars": longest,
        "return_convention": "additive",
    }


def period_metrics(performance, annualizer):
    return [
        {"year": int(y), **summarize(part, annualizer)}
        for y, part in performance.groupby(performance.index.year)
    ]


def with_equity(performance):
    p = performance.copy()
    p["equity"] = p["pnl"].cumsum()
    p["drawdown"] = p["equity"] - np.maximum.accumulate(np.r_[0.0, p["equity"]])[1:]
    return p
