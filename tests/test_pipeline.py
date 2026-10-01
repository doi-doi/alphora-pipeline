import json

import pytest

from alphora_pipeline.config import load_config
from alphora_pipeline.data import prepare, read_frame
from alphora_pipeline.demo import create_demo
from alphora_pipeline.pipeline import run
from alphora_pipeline.robustness import run_robustness
from alphora_pipeline.walk_forward import run_analysis


def test_complete_offline_pipeline_and_reproducible_diagnostics(tmp_path):
    config_path = create_demo(tmp_path / "inputs")
    run_dir = run(config_path, output_root=tmp_path / "runs")
    status = json.loads((run_dir / "status.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert status["status"] == "complete" and len(summary["folds"]) >= 4
    assert "report.html" in manifest["sha256"] and "raw/endpoint.parquet" in manifest["sha256"]
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "SYNTHETIC DEMONSTRATION" in report and "Plotly.newPlot" in report
    assert "cdn.plot.ly" not in report.split("</head>")[0]
    cfg, base = load_config(config_path)
    raw = {name: read_frame(base / spec["path"]) for name, spec in cfg["sources"].items()}
    raw["__price"] = read_frame(base / cfg["price"]["path"])
    frame, _ = prepare(cfg, raw)
    analysis = run_analysis(frame, cfg)
    a = run_robustness(frame, cfg, analysis)
    b = run_robustness(frame, cfg, analysis)
    assert a == b
    fee = [s for s in a["scenarios"] if s["scenario"] == "cost"]
    assert fee[0]["total_return"] >= fee[1]["total_return"] >= fee[2]["total_return"]


def test_failed_pipeline_retains_explicit_failure_status(tmp_path):
    config_path = create_demo(tmp_path / "inputs")
    cfg = json.loads(config_path.read_text())
    cfg["sources"]["endpoint"]["column"] = "does_not_exist"
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="expected columns"):
        run(config_path, output_root=tmp_path / "runs")
    status_files = list((tmp_path / "runs").glob("*/*/status.json"))
    assert len(status_files) == 1
    assert json.loads(status_files[0].read_text())["status"] == "failed"
