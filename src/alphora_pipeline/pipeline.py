"""Stage orchestration with immutable run directories and explicit failure status."""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .config import load_config
from .data import download, prepare
from .reporting import export, save_json
from .robustness import run_robustness
from .walk_forward import run_analysis


def run(config_path, output_root=None, cache_dir=None, refresh=False, env_file=None):
    cfg, base = load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    output_root = Path(output_root or root / "runs").resolve()
    cache_dir = Path(cache_dir or root / "cache").resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    run_dir = output_root / cfg["name"] / (stamp + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    log = logging.getLogger("alphora_pipeline")
    handler = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    try:
        save_json(run_dir / "resolved_config.json", cfg)
        save_json(run_dir / "status.json", {"status": "running", "stage": "download"})
        log.info("Acquiring source snapshots")
        raw, provenance = download(cfg, base, run_dir, cache_dir, refresh=refresh, env_file=env_file)
        save_json(run_dir / "status.json", {"status": "running", "stage": "cleaning"})
        frame, quality = prepare(cfg, raw)
        log.info("Cleaning complete: %d grid rows", len(frame))
        save_json(run_dir / "status.json", {"status": "running", "stage": "training_and_walk_forward"})
        analysis = run_analysis(frame, cfg)
        save_json(run_dir / "status.json", {"status": "running", "stage": "robustness"})
        robust = run_robustness(frame, cfg, analysis)
        save_json(run_dir / "status.json", {"status": "running", "stage": "reporting"})
        export(cfg, frame, quality, provenance, analysis, robust, run_dir)
        save_json(run_dir / "status.json", {"status": "complete", "report": "report.html"})
        log.info("Complete: %s", run_dir / "report.html")
        return run_dir
    except Exception as exc:
        # Keep the raw snapshots and configuration for diagnosis; never log credentials.
        save_json(run_dir / "status.json", {"status": "failed", "error_type": type(exc).__name__})
        log.error("Run failed (%s); artifacts retained in %s", type(exc).__name__, run_dir)
        raise
    finally:
        log.removeHandler(handler)
        handler.close()
