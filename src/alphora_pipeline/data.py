"""Immutable raw snapshots, auditable cleaning and causal source alignment."""

import asyncio
import hashlib
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import polars as pl
from dotenv import load_dotenv

from .config import timestamp


def read_frame(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path).to_pandas()
    raise ValueError("Input files must be CSV or Parquet")


def write_frame(df, path):
    pl.from_pandas(df.reset_index()).write_parquet(path)


def clean_series(raw, spec, cfg, name):
    time_col = spec.get("timestamp_column", "start_time")
    column = spec["column"]
    if time_col not in raw or column not in raw:
        raise ValueError(f"{name}: expected columns {time_col!r}, {column!r}; got {list(raw.columns)}")
    values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float, copy=True)
    unit = spec.get("timestamp_unit", "ms")
    times = pd.to_datetime(
        raw[time_col],
        unit=unit if pd.api.types.is_numeric_dtype(raw[time_col]) and unit != "iso" else None,
        utc=True,
        errors="coerce",
    )
    invalid_times = int(times.isna().sum())
    if invalid_times:
        raise ValueError(f"{name}: {invalid_times} invalid timestamps")
    invalid_values = int((~np.isfinite(values)).sum())
    values[~np.isfinite(values)] = np.nan
    s = pd.Series(values, index=pd.DatetimeIndex(times), name=name).sort_index(kind="stable")
    duplicates = int(s.index.duplicated().sum())
    if duplicates and cfg["cleaning"]["duplicates"] == "error":
        raise ValueError(f"{name}: {duplicates} duplicate timestamps; configure an explicit policy")
    if duplicates:
        s = s[~s.index.duplicated(keep=cfg["cleaning"]["duplicates"])]
    s.index = s.index + pd.Timedelta(minutes=spec.get("availability_lag_minutes", 0))
    return s, {
        "raw_rows": len(raw),
        "duplicates": duplicates,
        "invalid_values": invalid_values,
        "availability_lag_minutes": spec.get("availability_lag_minutes", 0),
    }


async def acquire(cfg, base, run_dir, cache_dir, refresh=False, env_file=None):
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    specs = {**cfg["sources"], "__price": cfg["price"]}
    live = [s for s in specs.values() if s.get("topic")]
    loader = None
    if live:
        api_key = os.getenv("CYBOTRADE_API_KEY", "")
        if not api_key:
            raise ValueError("Set CYBOTRADE_API_KEY in the project-root .env or use local source files")
        api_url = os.getenv("CYBOTRADE_API_URL", "").strip()
        if api_url and (urlsplit(api_url).scheme not in {"https", "http"} or not urlsplit(api_url).netloc):
            raise ValueError("CYBOTRADE_API_URL must be blank or a valid HTTP(S) URL")
        from adrs.data import DataLoader

        loader = DataLoader(
            data_dir=str(cache_dir),
            credentials={"cybotrade_api_key": api_key},
            cybotrade_api_url=api_url or None,
        )
    freq = pd.Timedelta(cfg["frequency"])
    begin = timestamp(cfg["start"]) - cfg["warmup_bars"] * freq
    finish = timestamp(cfg["end"]) + freq + pd.Timedelta(minutes=cfg["execution"]["price_shift_minutes"])
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True)
    result, provenance = {}, {}
    for name, spec in specs.items():
        if spec.get("topic"):
            # Serialize downloads: the SDK owns caching/retries and provider rate limits.
            frame = await loader.load(
                spec["topic"], begin.to_pydatetime(), finish.to_pydatetime(), override_existing=refresh
            )
            raw = frame.to_pandas()
        else:
            path = (base / spec["path"]).resolve()
            raw = read_frame(path)
        if raw.empty:
            raise ValueError(f"{name}: datasource returned no rows")
        path = raw_dir / (name + ".parquet")
        pl.from_pandas(raw).write_parquet(path)
        provenance[name] = {
            "source": spec,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "snapshot": path.relative_to(run_dir).as_posix(),
            "download_start": begin.isoformat(),
            "download_end_exclusive": finish.isoformat(),
        }
        result[name] = raw
    return result, provenance


def prepare(cfg, raw):
    freq = pd.Timedelta(cfg["frequency"])
    begin = timestamp(cfg["start"]) - cfg["warmup_bars"] * freq
    grid = pd.date_range(begin, timestamp(cfg["end"]), freq=freq, inclusive="left", name="start_time")
    frame = pd.DataFrame(index=grid)
    quality = {}
    for name, spec in cfg["sources"].items():
        s, record = clean_series(raw[name], spec, cfg, name)
        # Ceil availability onto the decision grid; never use a future observation.
        s.index = s.index.ceil(freq)
        if s.index.has_duplicates:
            raise ValueError(
                f"{name}: multiple observations align to one signal bar; explicitly aggregate upstream"
            )
        aligned = s.reindex(grid)
        record["missing_before_fill"] = int(aligned.isna().sum())
        if cfg["cleaning"]["missing"] == "ffill":
            aligned = aligned.ffill(limit=cfg["cleaning"]["ffill_limit_bars"])
        record["filled_rows"] = int(aligned.notna().sum() - s.reindex(grid).notna().sum())
        research = aligned.loc[aligned.index >= timestamp(cfg["start"])]
        record["research_missing_fraction"] = float(research.isna().mean())
        record["warmup_missing_rows"] = int(aligned.loc[aligned.index < timestamp(cfg["start"])].isna().sum())
        if record["research_missing_fraction"] > cfg["cleaning"]["max_missing_fraction"]:
            raise ValueError(
                f"{name}: research missing fraction {record['research_missing_fraction']:.2%} exceeds tolerance"
            )
        if cfg["cleaning"]["missing"] == "error" and research.isna().any():
            raise ValueError(f"{name}: missing research observations")
        frame[name] = aligned
        quality[name] = record
    prices, price_quality = clean_series(raw["__price"], cfg["price"], cfg, "price")
    # Unlike row shifts, explicit timestamp sampling cannot shift across missing minute candles.
    offset = pd.Timedelta(minutes=cfg["execution"]["price_shift_minutes"])
    if cfg["execution"]["price_sampling"] == "bar_last":
        offset += freq - pd.Timedelta(cfg["execution"]["price_frequency"])
    targets = grid + offset
    if (targets < grid).any():
        raise ValueError("Execution prices cannot precede decisions")
    aligned_price = prices.reindex(targets).to_numpy(dtype=float)
    required = grid >= timestamp(cfg["start"])
    bad = ~np.isfinite(aligned_price) | (aligned_price <= 0)
    if (bad & required).any():
        raise ValueError(
            f"Missing or nonpositive execution prices at {int((bad & required).sum())} research timestamps; prices are never filled"
        )
    frame["price"] = aligned_price
    frame["execution_time"] = targets
    price_quality["execution_offset_minutes"] = offset.total_seconds() / 60
    price_quality["research_missing_rows"] = int((bad & required).sum())
    quality["price"] = price_quality
    return frame, quality


def download(cfg, base, run_dir, cache_dir, **kwargs):
    return asyncio.run(acquire(cfg, base, run_dir, cache_dir, **kwargs))
