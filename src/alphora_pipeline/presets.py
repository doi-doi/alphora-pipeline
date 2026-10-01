"""Translate the five previously extracted report configurations into explicit graphs."""

from copy import deepcopy


def from_report_config(report):
    id = report["alpha_info"]["alpha_id"]
    if id not in {
        "OPT25D_OI_01_04",
        "OPT25D_OI_01_06",
        "OPT25D_OI_02_03",
        "OPT25D_OI_03_01",
        "OPT25D_OI_04_01",
    }:
        raise ValueError("Only the five verified report formula families are supported by this importer")
    steps = []

    def unary(name, op, source, **kwargs):
        steps.append({"name": name, "op": op, "input": source, **kwargs})
        return name

    def binary(name, op, a, b):
        steps.append({"name": name, "op": op, "inputs": [a, b]})
        return name

    if id in {"OPT25D_OI_01_04", "OPT25D_OI_01_06"}:
        skew, oi, lookback = (
            ("skew_3m", "oi_total", 2160) if id.endswith("01_04") else ("skew_6m", "oi_crypto_rel", 4320)
        )
        delta = unary("skew_delta", "diff", skew, periods=24)
        z = unary("skew_z", "zscore", delta, window=lookback)
        negative = unary("negative_skew", "negate", z)
        oi_log = unary("oi_log", "log", oi)
        growth = unary("oi_growth", "diff", oi_log, periods=24)
        gate = binary("oi_growing", "gt", growth, 0)
        output = binary("alpha_factor", "multiply", negative, gate)
    elif id.endswith("02_03"):
        z = unary("skew_z", "zscore", "skew_1m", window=720)
        negative = unary("negative_skew", "negate", z)
        oi_log = unary("oi_log", "log", "oi_bin_gn")
        oi_rank = unary("oi_rank", "percentile_rank", oi_log, window=720)
        output = binary("alpha_factor", "multiply", negative, oi_rank)
    elif id.endswith("03_01"):
        z = unary("skew_z", "zscore", "skew_1w", window=168)
        negative = unary("negative_skew", "negate", z)
        oi_log = unary("oi_log", "log", "oi_total")
        growth = unary("oi_growth", "diff", oi_log, periods=24)
        oi_z = unary("oi_z", "zscore", growth, window=168)
        strength = unary("oi_strength", "abs", oi_z)
        output = binary("alpha_factor", "multiply", negative, strength)
    else:
        z = unary("skew_z", "zscore", "skew_1w", window=168)
        negative = unary("negative_skew", "negate", z)
        oi_log = unary("oi_log", "log", "oi_crypto_rel")
        oi_rank = unary("oi_rank", "percentile_rank", oi_log, window=168)
        denominator = binary("denominator", "add", oi_rank, 0.25)
        output = binary("alpha_factor", "divide", negative, denominator)
    surfaces = report["parameter_sweep_surfaces"]
    cfg = {
        "name": id,
        "start": report["data_start"],
        "end": report["data_end_exclusive"],
        "sources": {
            name: {
                "topic": s["topic"],
                "column": s["column"],
                "availability_lag_minutes": max(0, s.get("shift_min", 0)),
            }
            for name, s in report["datasources"].items()
        },
        "price": {"topic": report["price_topic"], "column": "close"},
        "factor": {"steps": steps, "output": output},
        "grid": {
            "model": [report["alpha_info"]["model"]],
            "window": sorted({s["window"] for s in surfaces}),
            "threshold_1": sorted({x for s in surfaces for x in s["threshold_1"]}),
            "threshold_2": sorted({x for s in surfaces for x in s["threshold_2"]}),
        },
        "execution": {
            "fee_bps": report["fees_percent"] * 100,
            "price_shift_minutes": report["adrs_price_shift_minutes"],
        },
        "baseline_parameters": {**report["parameters"], "model": report["alpha_info"]["model"]},
        "notes": [
            "Original fixed parameters are supplied as a descriptive baseline. Defaults add warm-up, training eligibility, connected-plateau selection, execution-time purging and a three-month holdout; WF results therefore differ from the previous script.",
            "Verify actual provider publication lags. Zero configured lag is an assumption, not point-in-time data certification.",
        ],
    }
    if id.endswith("02_03"):
        cfg["notes"].append(
            "KNOWN NON-REPRODUCTION: the previous current-input reconstruction had 412 position mismatches against the original HTML; the raw-factor discrepancy and original percentilerank_meannorm implementation remain unresolved. This preset runs the documented reconstructed formula and must not be labelled exact reproduction."
        )
    return deepcopy(cfg)
