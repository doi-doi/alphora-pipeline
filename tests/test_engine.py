import numpy as np
import pandas as pd

from alphora_pipeline.engine import backtest
from alphora_pipeline.metrics import summarize
from alphora_pipeline.walk_forward import window


def test_turnover_cost_previous_position_and_flat_reset():
    ix = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    price = pd.Series([100.0, 110.0, 99.0, 108.9], index=ix)
    positions = pd.Series([1.0, 1.0, -1.0, 0.0], index=ix)
    p = backtest(price, positions, fee_bps=3.5)
    np.testing.assert_allclose(p.pnl, [-0.00035, 0.1, -0.1 - 0.0007, -0.1 - 0.00035], atol=1e-12)
    np.testing.assert_allclose(p.turnover, [1, 0, 2, 1])
    reset = backtest(price, positions, start=ix[2], fee_bps=3.5)
    np.testing.assert_allclose(reset.pnl, [-0.00035, -0.10035])
    m = summarize(p, 8760)
    assert m["trade_events"] == 3 and m["turnover_units"] == 4


def test_cached_training_slice_equals_direct_flat_start():
    ix = pd.date_range("2021-01-01", periods=7, freq="1h", tz="UTC")
    prices = pd.Series([100, 101, 102, 99, 101, 103, 102], index=ix)
    positions = pd.Series([1, 1, 1, -1, -1, 0, 1], index=ix)
    full = backtest(prices, positions, fee_bps=5, slippage_bps=2)
    a = window(full, ix[2], ix[6], 7)
    b = backtest(prices, positions, start=ix[2], end=ix[6], fee_bps=5, slippage_bps=2)
    pd.testing.assert_frame_equal(a, b)


def test_delay_never_reaches_across_fold_boundary():
    ix = pd.date_range("2021-01-01", periods=4, freq="1h", tz="UTC")
    result = backtest(
        pd.Series([100, 101, 102, 103], index=ix),
        pd.Series([1, 1, -1, -1], index=ix),
        start=ix[2],
        delay_bars=1,
    )
    assert result.position.tolist() == [0.0, -1.0]


def test_drawdown_includes_initial_zero_equity():
    ix = pd.date_range("2021-01-01", periods=2, freq="1h", tz="UTC")
    p = backtest(pd.Series([100, 100], index=ix), pd.Series([1, 1], index=ix), fee_bps=10)
    assert summarize(p, 8760)["max_drawdown"] == -0.001
