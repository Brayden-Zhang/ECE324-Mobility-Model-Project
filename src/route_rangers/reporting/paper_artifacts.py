"""Generate paper tables and metric snapshots from checked-in repository artifacts."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from route_rangers.config import PROJECT_ROOT


DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "foundation_downstream_report.json"
DEFAULT_FOUNDATION_DOC = PROJECT_ROOT / "docs" / "foundation_evals.md"
DEFAULT_UNITRAJ_CSV = (
    PROJECT_ROOT / "reports" / "unitraj_hmt_external_comparison_20260324.csv"
)
COMPUTE_SCRIPT_SPECS = (
    ("HMT full pretraining", PROJECT_ROOT / "scripts" / "slurm_train_hmt_nohash_full.sh"),
    ("HMT macro stage-2", PROJECT_ROOT / "scripts" / "slurm_train_hmt_stage2.sh"),
    ("Full benchmark sweep", PROJECT_ROOT / "scripts" / "slurm_benchmark_latest.sh"),
    ("Downstream task suite", PROJECT_ROOT / "scripts" / "slurm_neurips_master.sh"),
    ("External UniTraj train", PROJECT_ROOT / "scripts" / "slurm_unitraj_train_external.sh"),
    ("External UniTraj eval", PROJECT_ROOT / "scripts" / "slurm_unitraj_eval_external.sh"),
    ("UniTraj CSV comparison", PROJECT_ROOT / "scripts" / "slurm_unitraj_compare.sh"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_rows(path: Path) -> list[dict[str, Any]]:
    return _load_json(path).get("rows", [])


def _row_value(
    rows: list[dict[str, Any]],
    task: str,
    metric: str,
    field: str = "model_value",
) -> float | None:
    for row in rows:
        if row.get("task") == task and row.get("metric") == metric and row.get("status") == "ok":
            value = row.get(field)
            return float(value) if value is not None else None
    return None


def _section(text: str, title: str) -> str:
    pattern = rf"^### {re.escape(title)}\n(.*?)(?=^### |\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _series_from_line(block: str, label: str) -> dict[str, float]:
    match = re.search(rf"- {re.escape(label)}: (.+)", block)
    if not match:
        return {}
    out: dict[str, float] = {}
    for item in match.group(1).split(","):
        piece = item.strip()
        if not piece or ":" not in piece:
            continue
        key, value = piece.split(":", 1)
        out[key.strip()] = float(value.strip())
    return out


def _literal_line(block: str, label: str) -> dict[str, float]:
    match = re.search(rf"- {re.escape(label)}: (.+)", block)
    if not match:
        return {}
    return {
        str(k): float(v)
        for k, v in ast.literal_eval(match.group(1).strip()).items()
    }


def _single_value_line(block: str, label: str) -> float | None:
    match = re.search(rf"- {re.escape(label)}: ([0-9eE+.\-]+)", block)
    return float(match.group(1)) if match else None


def _parse_markdown_table(block: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in block.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _benchmark_rows(foundation_doc: Path) -> dict[str, dict[str, float]]:
    text = foundation_doc.read_text(encoding="utf-8")
    block = _section(text, "Benchmarks (random split, test)")
    table = _parse_markdown_table(block)
    out: dict[str, dict[str, float]] = {}
    for row in table:
        run = row.get("run")
        if not run:
            continue
        out[run] = {
            "recon_l1": float(row["recon@l1"]),
            "next_top1": float(row["next@top1"]),
            "dest_top1": float(row["dest@top1"]),
        }
    return out


def _read_unitraj_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pick_unitraj_rows(rows: list[dict[str, str]], prefix: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        model = row["model"]
        if model == prefix or model.startswith(prefix):
            out[row["task"]] = {
                "mae_m": float(row["mae_m"]),
                "rmse_m": float(row["rmse_m"]),
                "n": float(row["n"]),
            }
    return out


def _percent_gain(model_value: float, baseline_value: float) -> float:
    if baseline_value == 0:
        return 0.0
    return 100.0 * (baseline_value - model_value) / baseline_value


def _slurm_directives(path: Path) -> dict[str, str]:
    directives: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#SBATCH "):
            continue
        directive = line[len("#SBATCH ") :].strip()
        if "=" in directive:
            key, value = directive.split("=", 1)
            directives[key.strip()] = value.strip()
        else:
            directives[directive] = ""
    return directives


def _gpu_budget(gres: str | None) -> str:
    if not gres:
        return "CPU-only"
    parts = gres.split(":")
    if len(parts) >= 3 and parts[0] == "gpu":
        gpu_type = parts[1].upper()
        count = parts[2]
        return f"{count}x {gpu_type}"
    if len(parts) == 2 and parts[0] == "gpu":
        return f"1x {parts[1].upper()}"
    return gres


def _compute_resources() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage, script_path in COMPUTE_SCRIPT_SPECS:
        directives = _slurm_directives(script_path)
        rows.append(
            {
                "stage": stage,
                "gpu": _gpu_budget(directives.get("--gres")),
                "cpus": directives.get("--cpus-per-task", "--"),
                "memory": directives.get("--mem", "--"),
                "time": directives.get("--time", "--"),
            }
        )
    return rows


def build_paper_metrics(
    report_json: Path | None = None,
    foundation_doc: Path | None = None,
    unitraj_csv: Path | None = None,
) -> dict[str, Any]:
    report_json = report_json or DEFAULT_REPORT_JSON
    foundation_doc = foundation_doc or DEFAULT_FOUNDATION_DOC
    unitraj_csv = unitraj_csv or DEFAULT_UNITRAJ_CSV

    rows = _report_rows(report_json)
    foundation_text = foundation_doc.read_text(encoding="utf-8")

    invariance_block = _section(foundation_text, "Invariance suite")
    reverse_block = _section(foundation_text, "Reverse-order stress")
    change_block = _section(foundation_text, "Change detection")
    macro_block = _section(foundation_text, "Macro distribution head")
    length_block = _section(foundation_text, "Length sensitivity")

    unitraj_rows = _read_unitraj_csv(unitraj_csv)
    clean_hmt = _pick_unitraj_rows(unitraj_rows, "unitraj_eval_regression_best")
    robust_hmt = _pick_unitraj_rows(unitraj_rows, "unitraj_eval_regression_robust_best")
    baseline = _pick_unitraj_rows(unitraj_rows, "unitraj")

    length_long = _row_value(rows, "length_sensitivity", "long_dest_top1")
    length_gap = _row_value(rows, "length_sensitivity", "gap_dest_top1")
    length_short = (
        length_long - length_gap
        if length_long is not None and length_gap is not None
        else None
    )
    bins_match = re.search(r"bins:\s*\[([0-9]+),\s*([0-9]+)\]", length_block)
    bins = (
        [int(bins_match.group(1)), int(bins_match.group(2))]
        if bins_match
        else None
    )

    metrics = {
        "sources": {
            "report_json": str(report_json),
            "foundation_doc": str(foundation_doc),
            "unitraj_csv": str(unitraj_csv),
        },
        "benchmarks": _benchmark_rows(foundation_doc),
        "length": {
            "bins": bins,
            "short_dest_top1": length_short,
            "long_dest_top1": length_long,
            "gap_dest_top1": length_gap,
        },
        "invariance": {
            "prefix_dest_top1": _series_from_line(invariance_block, "prefix dest_top1"),
            "time_shift_dest_top1": _series_from_line(
                invariance_block, "time-shift dest_top1"
            ),
            "downsample_dest_top1": _series_from_line(
                invariance_block, "downsample dest_top1"
            ),
        },
        "reverse_order": {
            "original": _literal_line(reverse_block, "original"),
            "reversed": _literal_line(reverse_block, "reversed"),
            "delta": _literal_line(reverse_block, "delta"),
        },
        "change_detection": {
            "pos_mean_dist": _single_value_line(change_block, "pos_mean_dist"),
            "neg_mean_dist": _single_value_line(change_block, "neg_mean_dist"),
            "auc": _single_value_line(change_block, "auc"),
        },
        "embedding_retrieval": {
            "top1": _row_value(rows, "embedding_retrieval", "top1"),
            "top5": _row_value(rows, "embedding_retrieval", "top5"),
        },
        "macro": {
            "macro_kl": _row_value(rows, "macro_eval", "macro_kl"),
            "macro_js": _row_value(rows, "macro_eval", "macro_js"),
            "macro_l1": _row_value(rows, "macro_eval", "macro_l1"),
            "macro_top1": _row_value(rows, "macro_eval", "macro_top1"),
            "n": _single_value_line(macro_block, "n"),
        },
        "compute_resources": _compute_resources(),
        "unitraj_clean": {},
        "unitraj_robust": {},
    }

    for task in ("recovery", "prediction"):
        hmt = clean_hmt.get(task, {})
        ext = baseline.get(task, {})
        if hmt and ext:
            metrics["unitraj_clean"][task] = {
                "hmt_mae_m": hmt["mae_m"],
                "hmt_rmse_m": hmt["rmse_m"],
                "baseline_mae_m": ext["mae_m"],
                "baseline_rmse_m": ext["rmse_m"],
                "mae_gain_pct": _percent_gain(hmt["mae_m"], ext["mae_m"]),
                "rmse_gain_pct": _percent_gain(hmt["rmse_m"], ext["rmse_m"]),
            }
        robust = robust_hmt.get(task, {})
        if robust:
            metrics["unitraj_robust"][task] = {
                "hmt_mae_m": robust["mae_m"],
                "hmt_rmse_m": robust["rmse_m"],
            }
    return metrics


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _length_table(metrics: dict[str, Any]) -> str:
    length = metrics["length"]
    bins = length.get("bins")
    caption = "Destination-oriented length sensitivity from the checked-in LengthRobustBench artifact."
    if bins:
        caption = (
            f"{caption} Quantile thresholds are {bins[0]} and {bins[1]}. "
            "Short-bucket accuracy is recovered from the reported long-bucket score and LSG."
        )
    return f"""% Auto-generated by route_rangers.reporting.paper_artifacts
\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{tab:length}}
\\begin{{tabular}}{{lcc}}
\\toprule
Bucket / Summary & Dest-Local@Top1 & Value \\\\
\\midrule
Short & accuracy & {_fmt(length.get("short_dest_top1"))} \\\\
Long & accuracy & {_fmt(length.get("long_dest_top1"))} \\\\
LSG & long - short & {_fmt(length.get("gap_dest_top1"))} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def _robustness_table(metrics: dict[str, Any]) -> str:
    inv = metrics["invariance"]
    rev = metrics["reverse_order"]
    change = metrics["change_detection"]
    return f"""% Auto-generated by route_rangers.reporting.paper_artifacts
\\begin{{table*}}[t]
\\centering
\\caption{{LengthRobustBench robustness summary from the checked-in evaluation artifacts.}}
\\label{{tab:robustness}}
\\small
\\renewcommand{{\\arraystretch}}{{1.15}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\\toprule
\\textbf{{Perturbation Type}} & \\textbf{{Diagnostic Objective}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Observation Strata}}}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{Prefix Scaling Constraint}} & \\multirow{{2}}{{*}}{{Dest-Local@Top1 Accuracy $\\uparrow$}} & 25\\% Obs. & 50\\% Obs. & 75\\% Obs. & 100\\% Obs. \\\\
 & & {_fmt(inv["prefix_dest_top1"].get("0.25"))} & {_fmt(inv["prefix_dest_top1"].get("0.50"))} & {_fmt(inv["prefix_dest_top1"].get("0.75"))} & {_fmt(inv["prefix_dest_top1"].get("1.00"))} \\\\
\\cmidrule{{3-6}}
\\multirow{{2}}{{*}}{{Temporal Shift Distortion}} & \\multirow{{2}}{{*}}{{Dest-Local@Top1 Accuracy $\\uparrow$}} & 0 Hours & +12 Hours & +24 Hours & --- \\\\
 & & {_fmt(inv["time_shift_dest_top1"].get("0"))} & {_fmt(inv["time_shift_dest_top1"].get("43200"))} & {_fmt(inv["time_shift_dest_top1"].get("86400"))} & --- \\\\
\\cmidrule{{3-6}}
\\multirow{{2}}{{*}}{{Sparse Downsampling}} & \\multirow{{2}}{{*}}{{Dest-Local@Top1 Accuracy $\\uparrow$}} & 25\\% Kept & 50\\% Kept & --- & --- \\\\
 & & {_fmt(inv["downsample_dest_top1"].get("0.25"))} & {_fmt(inv["downsample_dest_top1"].get("0.50"))} & --- & --- \\\\
\\midrule
\\textbf{{Structural Integrity Test}} & \\textbf{{Metric Variants}} & \\textbf{{Top-1}} ($\\uparrow$) & \\textbf{{Top-5}} ($\\uparrow$) & \\textbf{{NLL}} ($\\downarrow$) & \\\\
\\midrule
Original Trajectory Anchor & Dest-Local Prediction Stats & {_fmt(rev["original"].get("top1"))} & {_fmt(rev["original"].get("top5"))} & {_fmt(rev["original"].get("nll"))} & \\\\
Reversed Temporal Order & Dest-Local Prediction Stats & {_fmt(rev["reversed"].get("top1"))} & {_fmt(rev["reversed"].get("top5"))} & {_fmt(rev["reversed"].get("nll"))} & \\\\
\\midrule
\\textbf{{Embedding Latent Space}} & \\textbf{{Change Detection Target}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Positive Dist}} ($\\downarrow$)}} & \\multicolumn{{2}}{{c}}{{\\textbf{{Negative Dist}} ($\\uparrow$)}} \\\\
Discriminative Boundary & AUC = \\textbf{{{_fmt(change.get("auc"))}}} ($\\uparrow$) & \\multicolumn{{2}}{{c}}{{{_fmt(change.get("pos_mean_dist"))}}} & \\multicolumn{{2}}{{c}}{{{_fmt(change.get("neg_mean_dist"))}}} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}"""


def _unitraj_main_table(metrics: dict[str, Any]) -> str:
    rec = metrics["unitraj_clean"]["recovery"]
    pred = metrics["unitraj_clean"]["prediction"]
    return f"""% Auto-generated by route_rangers.reporting.paper_artifacts
\\begin{{table}}[t]
\\centering
\\caption{{Main comparison against the UniTraj-compatible baseline under the shared evaluation wrapper.}}
\\label{{tab:unitraj_main}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Task / Model}} & \\textbf{{MAE (m)}} $\\downarrow$ & \\textbf{{RMSE (m)}} $\\downarrow$ & \\textbf{{MAE Improv.}} & \\textbf{{RMSE Improv.}} \\\\
\\midrule
\\textit{{Trajectory Recovery}} & & & & \\\\
UniTraj & {_fmt(rec["baseline_mae_m"], 2)} & {_fmt(rec["baseline_rmse_m"], 2)} & -- & -- \\\\
\\textbf{{HMT (Ours)}} & \\textbf{{{_fmt(rec["hmt_mae_m"], 2)}}} & \\textbf{{{_fmt(rec["hmt_rmse_m"], 2)}}} & \\textbf{{{_fmt(rec["mae_gain_pct"], 1)}\\%}} & \\textbf{{{_fmt(rec["rmse_gain_pct"], 1)}\\%}} \\\\
\\midrule
\\textit{{Next-Step Prediction}} & & & & \\\\
UniTraj & {_fmt(pred["baseline_mae_m"], 2)} & {_fmt(pred["baseline_rmse_m"], 2)} & -- & -- \\\\
\\textbf{{HMT (Ours)}} & \\textbf{{{_fmt(pred["hmt_mae_m"], 2)}}} & \\textbf{{{_fmt(pred["hmt_rmse_m"], 2)}}} & \\textbf{{{_fmt(pred["mae_gain_pct"], 1)}\\%}} & \\textbf{{{_fmt(pred["rmse_gain_pct"], 1)}\\%}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def _unitraj_appendix_table(metrics: dict[str, Any]) -> str:
    rec = metrics["unitraj_clean"]["recovery"]
    pred = metrics["unitraj_clean"]["prediction"]
    return f"""% Auto-generated by route_rangers.reporting.paper_artifacts
\\begin{{table}}[t]
\\centering
\\caption{{UniTraj-style clean evaluation (meters).}}
\\label{{tab:unitraj_clean}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Model & Rec MAE & Rec RMSE & Pred MAE & Pred RMSE \\\\
\\midrule
External UniTraj & {_fmt(rec["baseline_mae_m"], 2)} & {_fmt(rec["baseline_rmse_m"], 2)} & {_fmt(pred["baseline_mae_m"], 2)} & {_fmt(pred["baseline_rmse_m"], 2)} \\\\
HMT (regression) & {_fmt(rec["hmt_mae_m"], 2)} & {_fmt(rec["hmt_rmse_m"], 2)} & {_fmt(pred["hmt_mae_m"], 2)} & {_fmt(pred["hmt_rmse_m"], 2)} \\\\
Relative gain (\\%) & -{_fmt(rec["mae_gain_pct"], 1)} & -{_fmt(rec["rmse_gain_pct"], 1)} & -{_fmt(pred["mae_gain_pct"], 1)} & -{_fmt(pred["rmse_gain_pct"], 1)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def _macro_table(metrics: dict[str, Any]) -> str:
    macro = metrics["macro"]
    return f"""% Auto-generated by route_rangers.reporting.paper_artifacts
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Macro-head evaluation snapshot from the checked-in report artifact.}}
\\label{{tab:macro_compute}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Kullback-Leibler Divergence ($\\downarrow$) & {_fmt(macro.get("macro_kl"))} \\\\
Jensen-Shannon Divergence ($\\downarrow$) & {_fmt(macro.get("macro_js"))} \\\\
Continuous $\\ell_1$ Variation ($\\downarrow$) & {_fmt(macro.get("macro_l1"))} \\\\
Macro Top-1 Accuracy ($\\uparrow$) & {_fmt(macro.get("macro_top1"))} \\\\
Evaluation samples & {_fmt(macro.get("n"), 0)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def _compute_table(metrics: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated by route_rangers.reporting.paper_artifacts",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Requested SLURM resource budgets for the main training and evaluation entrypoints used by the paper artifact pipeline. These are scheduler requests encoded in the repository scripts, not measured runtime traces.}",
        "\\label{tab:compute_resources}",
        "\\begin{tabular}{p{0.34\\linewidth}cccc}",
        "\\toprule",
        "Stage & GPU & CPUs & RAM & Wall-time \\\\",
        "\\midrule",
    ]
    for row in metrics["compute_resources"]:
        lines.append(
            f'{row["stage"]} & {row["gpu"]} & {row["cpus"]} & {row["memory"]} & {row["time"]} \\\\'
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def write_paper_artifacts(
    output_dir: Path,
    report_json: Path | None = None,
    foundation_doc: Path | None = None,
    unitraj_csv: Path | None = None,
) -> dict[str, Any]:
    metrics = build_paper_metrics(report_json, foundation_doc, unitraj_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "paper_metrics.json", json.dumps(metrics, indent=2))
    _write(output_dir / "paper_length_table.tex", _length_table(metrics))
    _write(output_dir / "paper_robustness_table.tex", _robustness_table(metrics))
    _write(output_dir / "paper_unitraj_main.tex", _unitraj_main_table(metrics))
    _write(output_dir / "paper_unitraj_clean.tex", _unitraj_appendix_table(metrics))
    _write(output_dir / "paper_macro_table.tex", _macro_table(metrics))
    _write(output_dir / "paper_compute_table.tex", _compute_table(metrics))
    return metrics
