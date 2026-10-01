"""One-bar-held positions and turnover costs, with explicit flat fold resets."""

import numpy as np
import pandas as pd


def backtest(prices, positions, start=None, end=None, fee_bps=3.5, slippage_bps=0, delay_bars=0):
    if not isinstance(delay_bars, int) or delay_bars < 0:
        raise ValueError("delay_bars must be a nonnegative integer")
    ix = prices.index
    mask = np.ones(len(ix), dtype=bool)
    if start is not None:
        mask &= ix >= start
    if end is not None:
        mask &= ix < end
    price = prices.loc[mask]
    if price.empty or not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("Backtest window needs finite positive prices")
    # A delayed execution cannot reach backward across a fold reset.
    pos = positions.reindex(price.index).shift(delay_bars).fillna(0).astype(float)
    if not pos.isin([-1, 0, 1]).all():
        raise ValueError("Positions must be -1, 0 or 1")
    previous = pos.shift(1).fillna(0)
    returns = price.pct_change(fill_method=None).fillna(0)
    turnover = (pos - previous).abs()
    cost = turnover * (fee_bps + slippage_bps) / 10000
    gross = previous * returns
    return pd.DataFrame(
        {
            "price": price,
            "position": pos,
            "held_position": previous,
            "asset_return": returns,
            "turnover": turnover,
            "gross_pnl": gross,
            "cost": cost,
            "pnl": gross - cost,
        },
        index=price.index,
    )
