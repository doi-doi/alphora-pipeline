"""Self-contained interactive HTML and machine-readable research artifacts."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .data import write_frame


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def json_text(value, pretty=False):
    return json.dumps(clean(value), indent=2 if pretty else None, allow_nan=False)


def save_json(path, value):
    Path(path).write_text(json_text(value, True), encoding="utf-8")


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def fmt(value, kind="number"):
    if value is None or isinstance(value, float) and not np.isfinite(value):
        return "—"
    if isinstance(value, (dict, list)):
        return escape(json_text(value))
    if kind == "percent":
        return f"{value * 100:.2f}%"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    return escape(str(value))


def table(rows, columns):
    # columns: key, readable title, optional unit convention
    head = "".join(f"<th>{escape(c[1])}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{fmt(r.get(c[0]), c[2] if len(c) > 2 else 'number')}</td>" for c in columns)
        + "</tr>"
        for r in rows
    )
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    )


METRICS = [
    ("sharpe", "Sharpe"),
    ("sortino", "Sortino"),
    ("total_return", "Additive return", "percent"),
    ("annual_return", "Annualized additive return", "percent"),
    ("max_drawdown", "Maximum drawdown", "percent"),
    ("calmar", "Calmar"),
    ("annual_volatility", "Annual volatility", "percent"),
    ("exposure", "Exposure", "percent"),
    ("long_exposure", "Long exposure", "percent"),
    ("short_exposure", "Short exposure", "percent"),
    ("bar_win_rate", "Winning nonzero bars", "percent"),
    ("bar_profit_factor", "Bar profit factor"),
    ("trade_events", "Trade events"),
    ("turnover_units", "Turnover units"),
    ("fees_and_slippage", "Total cost", "percent"),
    ("longest_drawdown_bars", "Longest drawdown (bars)"),
    ("rows", "Hourly/signal bars"),
]


def comparison_table(metrics):
    rows = [
        {
            "metric": m[1],
            **{
                label: fmt(v.get(m[0]), m[2] if len(m) > 2 else "number") if v else "—"
                for label, v in metrics.items()
            },
        }
        for m in METRICS
    ]
    # Formatted metric cells only contain escaped text or our own numeric strings.
    header = "<th>Metric</th>" + "".join(f"<th>{escape(k)}</th>" for k in metrics)
    body = "".join(
        "<tr><td>" + r["metric"] + "</td>" + "".join("<td>" + r[k] + "</td>" for k in metrics) + "</tr>"
        for r in rows
    )
    return f'<div class="table-scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def chart_path(performance, title):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=["Additive equity (%)", "Drawdown (%)"],
    )
    daily = (
        performance[["equity", "drawdown"]].resample("1D").agg({"equity": "last", "drawdown": "min"}).dropna()
    )
    fig.add_trace(
        go.Scatter(x=daily.index, y=daily.equity * 100, mode="lines", name="Equity", line_color="#1466c3"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=daily.index, y=daily.drawdown * 100, mode="lines", name="Drawdown", line_color="#bd463a"
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title, template="plotly_white", height=620, margin={"l": 55, "r": 25, "t": 75, "b": 45}
    )
    return fig


def generate_html(cfg, quality, provenance, analysis, robust, run_dir):
    figures = []

    def chart(figure):
        id = f"chart-{len(figures)}"
        figures.append((id, figure.to_json()))
        return f'<div class="chart" id="{id}"></div>'

    tested = [f for f in analysis["folds"] if f["status"] == "tested"]
    skipped = [f for f in analysis["folds"] if f["status"] == "skipped"]
    rec = analysis["recommendation"]
    selected = rec["selected"]
    warnings = []
    if skipped:
        warnings.append(
            f"{len(skipped)} folds were skipped because no training candidate passed eligibility. OOS coverage is {analysis['coverage']:.1%}."
        )
    if selected.get("singleton_plateau"):
        warnings.append(
            "The final selected plateau contains only one grid point; local parameter stability is weak."
        )
    if selected["best_sharpe"] <= 0:
        warnings.append(
            "The best eligible final training Sharpe is nonpositive; the pipeline still reports the best available candidate without claiming an edge."
        )
    if cfg["walk_forward"]["holdout_months"] == 0:
        warnings.append(
            "No untouched holdout was reserved. The recommendation was fitted at the end of the historical data."
        )
    warnings.append(
        "Source observations are current downloaded snapshots. Configured availability lags do not recover historical provider revisions or prove point-in-time availability."
    )
    warnings.extend(cfg.get("notes", []))
    quality_rows = [{"source": k, **v} for k, v in quality.items()]
    folds = [
        {
            "fold": f["fold"],
            "status": f["status"],
            "test_start": f["test_start"],
            "test_end": f["test_end"],
            "purged_bars": f["purged_bars"],
            "parameters": f.get("selected", {}).get("parameters"),
            "train_sharpe": f.get("selected", {}).get("sharpe"),
            "plateau_size": f.get("selected", {}).get("plateau_size"),
            "test_sharpe": f.get("test_metrics", {}).get("sharpe"),
            "test_return": f.get("test_metrics", {}).get("total_return"),
            "test_mdd": f.get("test_metrics", {}).get("max_drawdown"),
            "reason": f.get("reason", ""),
        }
        for f in analysis["folds"]
    ]
    fold_fig = go.Figure()
    fold_fig.add_bar(
        x=[f["fold"] for f in folds], y=[f["train_sharpe"] for f in folds], name="Training Sharpe"
    )
    fold_fig.add_bar(x=[f["fold"] for f in folds], y=[f["test_sharpe"] for f in folds], name="Test Sharpe")
    fold_fig.update_layout(
        title="Training versus untouched fold tests",
        template="plotly_white",
        barmode="group",
        height=340,
        xaxis_title="Fold",
        yaxis_title="Sharpe",
    )
    score_rows = analysis["training_scores"][-1]
    heat_rows = [
        r
        for r in score_rows
        if r["parameters"]["model"] == selected["parameters"]["model"]
        and r["parameters"]["threshold_2"] == selected["parameters"]["threshold_2"]
        and all(
            r["parameters"][k] == selected["parameters"][k]
            for k in cfg["grid"]
            if k not in {"model", "window", "threshold_1", "threshold_2"}
        )
    ]
    heat = pd.DataFrame(
        [
            {
                "window": r["parameters"]["window"],
                "long": r["parameters"]["threshold_1"],
                "sharpe": r["sharpe"] if r["eligible"] else np.nan,
            }
            for r in heat_rows
        ]
    ).pivot(index="window", columns="long", values="sharpe")
    heat_fig = go.Figure(
        go.Heatmap(
            x=heat.columns,
            y=heat.index,
            z=heat.values,
            colorscale="RdBu",
            colorbar_title="Train SR",
            hovertemplate="Window %{y}<br>Long %{x}<br>Training Sharpe %{z:.4f}<extra></extra>",
        )
    )
    heat_fig.update_layout(
        title="Final training-only Sharpe surface (other parameters fixed to recommendation)",
        template="plotly_white",
        height=380,
        xaxis_title="Long threshold",
        yaxis_title="Window",
    )
    bootstrap_rows = [{"metric": k, **v} for k, v in robust["bootstrap"]["intervals"].items()]
    snapshots = [
        {
            "source": k,
            "topic_or_file": v["source"].get("topic", v["source"].get("path")),
            "column": v["source"]["column"],
            "sha256": v["sha256"],
        }
        for k, v in provenance.items()
    ]
    holdout = (
        chart(chart_path(analysis["holdout"], "Final untouched holdout"))
        if analysis["holdout"] is not None
        else "<p>No holdout configured.</p>"
    )
    content = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(cfg["name"])} — research report</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#193047;background:#f2f5f8;margin:0}}main{{max-width:1240px;margin:auto;padding:36px}}section,.summary{{background:white;border:1px solid #dbe4ec;border-radius:12px;padding:24px;margin:24px 0}}h1{{font-size:32px}}h2{{font-size:23px}}h3{{font-size:18px}}nav{{display:flex;flex-wrap:wrap;gap:18px}}a{{color:#1466c3}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #e2e9ee;text-align:right;vertical-align:top}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf3f8}}.table-scroll{{overflow:auto}}.chart{{min-height:340px}}.warning{{background:#fff5da;padding:16px;border-radius:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f7;padding:16px;font-size:13px}}section{{scroll-margin-top:16px}}@media(max-width:700px){{main{{padding:12px}}section{{padding:12px}}h1{{font-size:25px}}}}@media print{{body{{background:white}}section{{break-inside:avoid}}nav{{display:none}}}}</style></head><body><main>
<h1>{escape(cfg["name"])} — endpoint research report</h1><p>Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}. All timestamps UTC. Returns use the additive convention from the previous backtests.</p>
<nav><a href="#recommendation">Parameters</a><a href="#metrics">Metrics</a><a href="#wfa">Walk-forward</a><a href="#holdout">Holdout</a><a href="#robustness">Robustness</a><a href="#data">Data & methodology</a></nav>
<div class="summary"><h2>Result</h2><p>{len(tested)} tested folds; {len(skipped)} skipped folds; coverage {analysis["coverage"]:.1%}. Walk-forward Sharpe {fmt(analysis["oos_metrics"]["sharpe"])}, return {fmt(analysis["oos_metrics"]["total_return"], "percent")}, drawdown {fmt(analysis["oos_metrics"]["max_drawdown"], "percent")}.</p><div class="warning">{"<br>".join(escape(w) for w in warnings)}</div></div>
<section id="recommendation"><h2>Recommended research parameters</h2><pre>{escape(json_text(selected["parameters"], True))}</pre><p>Selected using only {escape(str(rec["train_start"]))} through {escape(str(rec["train_end"]))} (exclusive), before {escape(str(rec["parameters_as_of"]))}. Selection: {escape(selected["selection"])}; plateau size {selected["plateau_size"]}; selected train Sharpe {fmt(selected["sharpe"])}; best eligible train Sharpe {fmt(selected["best_sharpe"])}. This is an estimated research candidate, not a guarantee of optimal future performance.</p>{chart(heat_fig)}</section>
<section id="metrics"><h2>Relevant metrics</h2><p>Each column has its own date range. The descriptive full-history curve applies the final recommended parameters retrospectively; it is not an OOS performance claim. Bar win rate and bar profit factor measure PnL bars, not closed round-trip trades. A reversal is one trade event but two turnover units.</p>{comparison_table({"Walk-forward OOS": analysis["oos_metrics"], "Untouched holdout": analysis["holdout_metrics"], "Final parameters, descriptive full history": analysis["descriptive_metrics"], **({"Original fixed baseline": analysis["baseline_metrics"]} if analysis["baseline_metrics"] else {})})}<pre>{escape(json_text({"oos_dates": {"start": analysis["oos_metrics"]["start"], "end": analysis["oos_metrics"]["end_inclusive"]}, "holdout_dates": {"start": analysis["holdout_metrics"]["start"], "end": analysis["holdout_metrics"]["end_inclusive"]} if analysis["holdout_metrics"] else None}, True))}</pre></section>
<section id="wfa"><h2>Walk-forward analysis</h2><p>Each test starts flat. Parameters are selected only from the preceding {cfg["walk_forward"]["train_months"]} months; execution-time purging and {cfg["walk_forward"]["embargo_bars"]} embargo bars prevent training from consuming test-period prices. Skipped folds create coverage gaps and are never silently filled with zero PnL.</p>{chart(chart_path(analysis["oos"], "Stitched walk-forward out-of-sample path"))}{chart(fold_fig)}{table(folds, [("fold", "Fold"), ("status", "Status"), ("test_start", "Test start"), ("test_end", "Test end exclusive"), ("parameters", "Selected parameters"), ("train_sharpe", "Train SR"), ("test_sharpe", "Test SR"), ("test_return", "Test return", "percent"), ("test_mdd", "Test MDD", "percent"), ("plateau_size", "Plateau size"), ("purged_bars", "Purged bars"), ("reason", "Reason")])}</section>
<section id="holdout"><h2>Untouched final holdout</h2><p>The recommendation is frozen before the holdout begins. Holdout scores and robustness diagnostics never feed parameter selection. Repeated runs after inspecting this holdout turn it into research data; reserve new dates for a new honest test.</p>{holdout}</section>
<section id="robustness"><h2>Robustness diagnostics</h2><h3>Cost, slippage and delayed execution</h3><p>All stress runs reuse the original selected fold parameters. Slippage is a per-turnover sensitivity assumption; market impact and funding are not modelled.</p>{table(robust["scenarios"], [("scenario", "Scenario"), ("fee_multiplier", "Fee multiplier"), ("slippage_bps", "Slippage bps"), ("delay_bars", "Delay bars"), ("sharpe", "Sharpe"), ("total_return", "Return", "percent"), ("max_drawdown", "MDD", "percent")])}<h3>Training parameter neighbors</h3>{table(robust["parameter_neighbors"], [("parameters", "Parameters"), ("is_selected", "Selected"), ("sharpe", "Train Sharpe"), ("total_return", "Train return", "percent"), ("max_drawdown", "Train MDD", "percent")])}<h3>Conditional block bootstrap</h3><p>{escape(robust["bootstrap"]["interpretation"])} {robust["bootstrap"]["samples"]} samples, {robust["bootstrap"]["block_bars"]}-bar blocks. Return and drawdown below are decimal units.</p>{table(bootstrap_rows, [("metric", "Metric"), ("p025", "2.5%"), ("median", "Median"), ("p975", "97.5%")])}<h3>Conditional return-shift placebo and parameter stability</h3><pre>{escape(json_text({"placebo": robust["placebo"], "parameter_stability": robust["parameter_stability"]}, True))}</pre><h3>OOS calendar periods</h3>{table(robust["yearly_oos"], [("year", "Year"), ("sharpe", "Sharpe"), ("total_return", "Return", "percent"), ("max_drawdown", "MDD", "percent"), ("rows", "Bars")])}</section>
<section id="data"><h2>Data cleaning, sources and reproducibility</h2><p>No backward filling, globally fitted normalization, interpolation through future data, or automatic removal of market outliers. Missing prices fail. Factor nonfinite values are audited; rolling transformations and configured filling use past observations only. Source timestamps must describe observations whose availability is represented accurately by the configured lag.</p>{table(quality_rows, [("source", "Source"), ("raw_rows", "Raw rows"), ("duplicates", "Duplicates"), ("invalid_values", "Nonfinite values"), ("filled_rows", "Filled rows"), ("research_missing_fraction", "Research missing", "percent"), ("warmup_missing_rows", "Missing warm-up"), ("execution_offset_minutes", "Execution offset (min)")])}{table(snapshots, [("source", "Source"), ("topic_or_file", "Endpoint / file"), ("column", "Column"), ("sha256", "Raw snapshot SHA-256")])}<details><summary>Full resolved strategy configuration</summary><pre>{escape(json_text(cfg, True))}</pre></details><p>Metrics and artifacts: summary.json, recommended_parameters.json, fold_metrics.csv, training_grid.csv, robustness.json, quality.json, provenance.json, manifest.json, cleaned.parquet, walk_forward.parquet, holdout.parquet (if configured), descriptive_full_history.parquet, and baseline.parquet (if supplied).</p></section>
</main><script>{get_plotlyjs()}</script><script>"""
    for id, figure in figures:
        # Plotly JSON remains data even if a source title contains HTML/script characters.
        serialized = figure.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        content += f"{{const f={serialized};Plotly.newPlot('{id}',f.data,f.layout,{{responsive:true,displaylogo:false}});}}\n"
    content += "</script></body></html>"
    (run_dir / "report.html").write_text(content, encoding="utf-8")


def export(cfg, frame, quality, provenance, analysis, robust, run_dir):
    save_json(run_dir / "resolved_config.json", cfg)
    save_json(run_dir / "quality.json", quality)
    save_json(run_dir / "provenance.json", provenance)
    save_json(run_dir / "robustness.json", robust)
    summary = {
        "name": cfg["name"],
        "oos": analysis["oos_metrics"],
        "coverage": analysis["coverage"],
        "holdout": analysis["holdout_metrics"],
        "descriptive_full_history": analysis["descriptive_metrics"],
        "baseline": analysis["baseline_metrics"],
        "folds": analysis["folds"],
        "recommendation": analysis["recommendation"],
    }
    save_json(run_dir / "summary.json", summary)
    save_json(run_dir / "recommended_parameters.json", analysis["recommendation"])
    for name, df in [
        ("cleaned", frame),
        ("walk_forward", analysis["oos"]),
        ("holdout", analysis["holdout"]),
        ("descriptive_full_history", analysis["descriptive"]),
        ("baseline", analysis["baseline"]),
    ]:
        if df is not None:
            write_frame(df, run_dir / (name + ".parquet"))
    scores = [
        {
            "fold": w["fold"],
            "train_start": w["train_start"],
            "train_end": w["train_end"],
            **r["parameters"],
            **{k: v for k, v in r.items() if k not in {"parameters", "reasons"}},
            "reasons": ";".join(r["reasons"]),
        }
        for w, rs in zip(analysis["training_windows"], analysis["training_scores"], strict=True)
        for r in rs
    ]
    pd.DataFrame(scores).to_csv(run_dir / "training_grid.csv", index=False)
    folds = []
    for f in analysis["folds"]:
        folds.append(
            {
                "fold": f["fold"],
                "status": f["status"],
                "train_start": f["train_start"],
                "train_end": f["train_end"],
                "test_start": f["test_start"],
                "test_end": f["test_end"],
                "purged_bars": f["purged_bars"],
                **f.get("selected", {}).get("parameters", {}),
                "train_sharpe": f.get("selected", {}).get("sharpe"),
                **{"test_" + k: v for k, v in f.get("test_metrics", {}).items()},
                "reason": f.get("reason", ""),
            }
        )
    pd.DataFrame(folds).to_csv(run_dir / "fold_metrics.csv", index=False)
    generate_html(cfg, quality, provenance, analysis, robust, run_dir)
    versions = {m: importlib.metadata.version(m) for m in ["numpy", "pandas", "polars", "plotly"]}
    files = {
        p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run_dir.rglob("*")
        if p.is_file() and p.name not in {"manifest.json", "pipeline.log", "status.json"}
    }
    save_json(
        run_dir / "manifest.json",
        {
            "created_at": datetime.now(UTC),
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "dependencies": versions,
            "sha256": files,
        },
    )
