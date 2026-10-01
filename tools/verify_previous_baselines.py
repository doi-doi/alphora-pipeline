"""Manual regression against saved backtest_wf outputs; requires no API key.

Usage: python tools/verify_previous_baselines.py --results <previous results folder>
The fifth match establishes parity with the prior reconstruction, not its original HTML.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from alphora_pipeline.config import load_config, timestamp
from alphora_pipeline.engine import backtest
from alphora_pipeline.reporting import save_json
from alphora_pipeline.transforms import SignalFactory, factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/previous_baseline_verification.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = []
    for path in sorted(args.results.glob("OPT25D_OI_*/grid_search")):
        id = path.parent.name
        cfg, _ = load_config(root / "configs" / (id + ".json"))
        frame = pl.read_parquet(path / "factor_inputs.parquet").to_pandas().set_index("start_time")
        original = pl.read_parquet(path / "full_period.parquet").to_pandas().set_index("start_time")
        frame.index = pd.to_datetime(frame.index, utc=True)
        original.index = pd.to_datetime(original.index, utc=True)
        p = cfg["baseline_parameters"]
        actual_factor = factor(frame, cfg["factor"], p)
        expected_factor = frame["factor"]
        np.testing.assert_allclose(actual_factor, expected_factor, atol=1e-8, rtol=1e-8, equal_nan=True)
        positions, _ = SignalFactory(frame, cfg).positions(p)
        actual = backtest(
            original["price"],
            positions,
            timestamp(cfg["start"]),
            timestamp(cfg["end"]),
            cfg["execution"]["fee_bps"],
        )
        mismatch = int((actual["position"] != original["signal"]).sum())
        max_pnl = float((actual["pnl"] - original["pnl"]).abs().max())
        assert mismatch == 0 and max_pnl < 1e-10, (id, mismatch, max_pnl)
        checks.append(
            {
                "strategy": id,
                "rows": len(actual),
                "position_mismatches": mismatch,
                "max_pnl_difference": max_pnl,
                "comparison": "Previous backtest_wf saved baseline; 02_03 remains a non-reproduction of original HTML",
            }
        )
        print(id, "positions match; max PnL difference", max_pnl)
    if len(checks) != 5:
        raise ValueError("Expected saved results for all five reports")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, checks)


if __name__ == "__main__":
    main()
