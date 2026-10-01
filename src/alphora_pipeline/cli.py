"""Console entry point and configuration generation."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import combinations, load_config
from .demo import create_demo
from .pipeline import run
from .presets import from_report_config
from .reporting import save_json


def parser():
    p = argparse.ArgumentParser(description="Alphora endpoint backtesting pipeline")
    commands = p.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run", help="Download/clean, optimize, walk forward, stress test and report")
    r.add_argument("--config", type=Path, required=True)
    r.add_argument("--output", type=Path)
    r.add_argument("--cache", type=Path)
    r.add_argument("--env-file", type=Path)
    r.add_argument("--refresh", action="store_true", help="Refresh the SDK's cached endpoint snapshots")
    r.add_argument("--open", action="store_true", help="Open the completed HTML in your browser")
    v = commands.add_parser("validate", help="Validate configuration without network access")
    v.add_argument("--config", type=Path, required=True)
    d = commands.add_parser(
        "demo", help="Create clearly labelled synthetic data and run the complete pipeline offline"
    )
    d.add_argument("--directory", type=Path, default=Path("data/demo"))
    d.add_argument("--output", type=Path)
    d.add_argument("--open", action="store_true")
    i = commands.add_parser(
        "init", help="Create a one-endpoint strategy config; edit transformation and grid afterwards"
    )
    i.add_argument("--topic", required=True)
    i.add_argument("--column", default="v")
    i.add_argument("--name", default="my_strategy")
    i.add_argument("--start", default="2021-01-01T00:00:00Z")
    i.add_argument("--end", required=True)
    i.add_argument("--output", type=Path, default=Path("configs/my_strategy.json"))
    c = commands.add_parser(
        "import-report-config",
        help="Convert an extracted JSON configuration from one of the five previous reports",
    )
    c.add_argument("--source", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "import-report-config":
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        cfg = from_report_config(json.loads(args.source.read_text(encoding="utf-8")))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, cfg)
        load_config(args.output)
        print(f"Created {args.output.resolve()}")
        return 0
    if args.command == "init":
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        cfg = {
            "name": args.name,
            "start": args.start,
            "end": args.end,
            "sources": {
                "endpoint": {"topic": args.topic, "column": args.column, "availability_lag_minutes": 0}
            },
            "price": {"topic": "bybit-linear|candle?symbol=BTCUSDT&interval=1m", "column": "close"},
            "factor": {"steps": [], "output": "endpoint"},
            "grid": {
                "model": ["zscore"],
                "window": [60, 100, 140, 180],
                "threshold_1": [0.4, 0.8, 1.2, 1.6, 2.0],
                "threshold_2": [-2.0, -1.6, -1.2, -0.8, -0.4],
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, cfg)
        load_config(args.output)
        print(
            f"Created {args.output.resolve()}. Review availability lags, formula, grid and dates before running."
        )
        return 0
    if args.command == "validate":
        cfg, _ = load_config(args.config)
        print(f"Valid: {cfg['name']}; {sum(1 for _ in combinations(cfg))} candidates; no data downloaded")
        return 0
    if args.command == "demo":
        config_path = create_demo(args.directory)
        directory = run(config_path, output_root=args.output)
    else:
        directory = run(
            args.config,
            output_root=args.output,
            cache_dir=args.cache,
            refresh=args.refresh,
            env_file=args.env_file,
        )
    print(f"Report: {directory / 'report.html'}")
    print(f"Parameters: {directory / 'recommended_parameters.json'}")
    if args.open:
        import webbrowser

        webbrowser.open((directory / "report.html").as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
