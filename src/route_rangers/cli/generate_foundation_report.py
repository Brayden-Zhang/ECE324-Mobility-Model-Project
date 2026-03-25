#!/usr/bin/env python3
"""Generate a unified foundation-model downstream report from cache artifacts."""

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class MetricRow:
    task: str
    metric: str
    direction: str
    model_value: Optional[float]
    baseline_value: Optional[float]
    delta: Optional[float]
    model_artifact: str
    baseline_artifact: str
    status: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a downstream foundation-model report with baseline deltas"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--output_dir", type=str, default="reports")
    parser.add_argument(
        "--filename_prefix", type=str, default="foundation_downstream_report"
    )
    return parser.parse_args()


def latest_files(pattern: str) -> List[str]:
    files = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    files.sort(key=lambda p: os.path.getmtime(p))
    return files


def latest_match(
    cache_dir: str,
    pattern: str,
    include_substr: Optional[str] = None,
    exclude_substr: Optional[List[str]] = None,
) -> Optional[str]:
    files = latest_files(os.path.join(cache_dir, pattern))
    if include_substr:
        files = [f for f in files if include_substr in Path(f).name]
    if exclude_substr:
        files = [f for f in files if all(s not in Path(f).name for s in exclude_substr)]
    return files[-1] if files else None


def safe_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def pick_split(obj: dict) -> dict:
    splits = obj.get("splits") or {}
    for k in ("all", "random", "temporal"):
        if k in splits:
            return splits[k]
    if splits:
        return splits[next(iter(splits.keys()))]
    return {}


def as_float(x) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def compute_delta(model_v: Optional[float], base_v: Optional[float], direction: str):
    if model_v is None or base_v is None:
        return None
    if direction == "higher":
        return model_v - base_v
    return base_v - model_v


def extract_benchmark(obj: dict) -> Dict[str, Optional[float]]:
    split = pick_split(obj)
    return {
        "recon_acc_l0": as_float(split.get("reconstruction", {}).get("recon_acc_l0")),
        "recon_acc_l1": as_float(split.get("reconstruction", {}).get("recon_acc_l1")),
        "recon_acc_l2": as_float(split.get("reconstruction", {}).get("recon_acc_l2")),
        "next_top1": as_float(split.get("next_location_probe", {}).get("test", {}).get("top1")),
        "next_top5": as_float(split.get("next_location_probe", {}).get("test", {}).get("top5")),
        "dest_top1": as_float(split.get("destination_probe", {}).get("test", {}).get("top1")),
        "dest_top5": as_float(split.get("destination_probe", {}).get("test", {}).get("top5")),
        "next_reg_mae_m": as_float(
            split.get("next_location_regression_probe", {}).get("test", {}).get("mae_m")
        ),
        "dest_reg_mae_m": as_float(
            split.get("destination_regression_probe", {}).get("test", {}).get("mae_m")
        ),
    }


def extract_embedding_retrieval(obj: dict) -> Dict[str, Optional[float]]:
    return {"top1": as_float(obj.get("top1")), "top5": as_float(obj.get("top5"))}


def extract_change_detection(obj: dict) -> Dict[str, Optional[float]]:
    return {
        "auc": as_float(obj.get("auc")),
        "margin": (
            as_float(obj.get("neg_mean_dist")) - as_float(obj.get("pos_mean_dist"))
            if as_float(obj.get("neg_mean_dist")) is not None
            and as_float(obj.get("pos_mean_dist")) is not None
            else None
        ),
    }


def extract_reverse_order(obj: dict) -> Dict[str, Optional[float]]:
    delta = obj.get("delta", {})
    return {
        "delta_top1": as_float(delta.get("top1")),
        "delta_nll": as_float(delta.get("nll")),
    }


def extract_invariance(obj: dict) -> Dict[str, Optional[float]]:
    prefix = obj.get("prefix_destination", {})
    down = obj.get("downsample_destination", {})
    p100 = as_float(prefix.get("1.00", {}).get("dest_top1"))
    d025 = as_float(down.get("0.25", {}).get("dest_top1"))
    return {"prefix_1.00_dest_top1": p100, "downsample_0.25_dest_top1": d025}


def extract_length(obj: dict) -> Dict[str, Optional[float]]:
    metrics = obj.get("metrics", {})
    gap = obj.get("length_sensitivity_gap", {})
    return {
        "long_dest_top1": as_float(metrics.get("long", {}).get("dest_top1")),
        "gap_dest_top1": as_float(gap.get("dest_top1")),
    }


def extract_travel_time(obj: dict) -> Dict[str, Optional[float]]:
    ratios = obj.get("prefix_ratios", {})
    target = ratios.get("0.5") or ratios.get("0.50") or {}
    test = target.get("test", target)
    return {
        "mae_s@0.5": as_float(test.get("mae_s", test.get("mae_seconds"))),
        "rmse_s@0.5": as_float(test.get("rmse_s", test.get("rmse_seconds"))),
        "mape@0.5": as_float(test.get("mape")),
    }


def extract_anomaly(obj: dict) -> Dict[str, Optional[float]]:
    combined = obj.get("combined", {})
    return {
        "combined_auroc": as_float(combined.get("auroc")),
        "combined_pr_auc": as_float(combined.get("pr_auc")),
    }


def extract_trip_classification(obj: dict) -> Dict[str, Optional[float]]:
    tasks = obj.get("tasks", {})
    out = {}
    for k in ("speed_mode", "duration_bucket", "distance_bucket"):
        out[f"{k}_top1"] = as_float(tasks.get(k, {}).get("test", {}).get("top1"))
    return out


def extract_similarity(obj: dict) -> Dict[str, Optional[float]]:
    geo = obj.get("geographic_consistency", {})
    self_r = obj.get("self_retrieval", {})
    return {
        "self_retrieval_top1": as_float(self_r.get("self_retrieval_top1")),
        "self_retrieval_top10": as_float(self_r.get("self_retrieval_top10")),
        "origin_mean_dist_km": as_float(geo.get("origin_mean_dist_km")),
        "dest_mean_dist_km": as_float(geo.get("dest_mean_dist_km")),
        "spatial_mean_dist_km": as_float(geo.get("spatial_mean_dist_km")),
    }


def extract_macro_eval(obj: dict) -> Dict[str, Optional[float]]:
    metrics = obj.get("metrics", {})
    return {
        "macro_kl": as_float(metrics.get("macro_kl")),
        "macro_js": as_float(metrics.get("macro_js")),
        "macro_l1": as_float(metrics.get("macro_l1")),
        "macro_top1": as_float(metrics.get("macro_top1")),
    }


def extract_transfer_suite(obj: dict) -> Dict[str, Optional[float]]:
    datasets = obj.get("datasets", {})
    recovery_vals: List[float] = []
    prediction_vals: List[float] = []

    for dataset_obj in datasets.values():
        split = pick_split(dataset_obj)
        rec = as_float(split.get("recovery", {}).get("test", {}).get("mae_m"))
        pred = as_float(split.get("prediction", {}).get("test", {}).get("mae_m"))
        if rec is not None:
            recovery_vals.append(rec)
        if pred is not None:
            prediction_vals.append(pred)

    return {
        "recovery_mae_m_mean": (
            sum(recovery_vals) / len(recovery_vals) if recovery_vals else None
        ),
        "prediction_mae_m_mean": (
            sum(prediction_vals) / len(prediction_vals) if prediction_vals else None
        ),
    }


def extract_next_poi_eval(obj: dict) -> Dict[str, Optional[float]]:
    res = obj.get("results", {})
    test = res.get("next_poi", {}).get("test", {})
    uid = res.get("user_identification", {}).get("test", {})
    return {
        "next_poi_top1": as_float(test.get("top1")),
        "next_poi_top5": as_float(test.get("top5")),
        "next_poi_top10": as_float(test.get("top10")),
        "next_poi_ndcg10": as_float(test.get("ndcg10")),
        "next_poi_mrr": as_float(test.get("mrr")),
        "user_id_top1": as_float(uid.get("top1")),
    }


def extract_cross_city_transfer(obj: dict) -> Dict[str, Optional[float]]:
    agg = obj.get("aggregate", {})
    np_agg = agg.get("next_poi", {})
    uid_agg = agg.get("user_identification", {})
    return {
        "cross_city_top1_mean": as_float(np_agg.get("top1", {}).get("mean")),
        "cross_city_top5_mean": as_float(np_agg.get("top5", {}).get("mean")),
        "cross_city_top10_mean": as_float(np_agg.get("top10", {}).get("mean")),
        "cross_city_ndcg10_mean": as_float(np_agg.get("ndcg10", {}).get("mean")),
        "cross_city_mrr_mean": as_float(np_agg.get("mrr", {}).get("mean")),
        "cross_city_user_id_top1_mean": as_float(uid_agg.get("top1", {}).get("mean")),
    }


def extract_unitraj_regression(obj: dict) -> Dict[str, Optional[float]]:
    split = pick_split(obj)
    rec = split.get("recovery", {}).get("test", {})
    pred = split.get("prediction", {}).get("test", {})
    return {
        "recovery_mae_m": as_float(rec.get("mae_m")),
        "prediction_mae_m": as_float(pred.get("mae_m")),
    }


def extract_external_unitraj(obj: dict) -> Dict[str, Optional[float]]:
    metrics = obj.get("metrics", {})
    return {
        "recovery_mae_m": as_float(metrics.get("recovery", {}).get("mae_m")),
        "prediction_mae_m": as_float(metrics.get("prediction", {}).get("mae_m")),
    }


def make_rows(
    task: str,
    direction_by_metric: Dict[str, str],
    model_path: Optional[str],
    baseline_path: Optional[str],
    model_extractor: Callable[[dict], Dict[str, Optional[float]]],
    baseline_extractor: Optional[Callable[[dict], Dict[str, Optional[float]]]] = None,
) -> List[MetricRow]:
    rows: List[MetricRow] = []
    model_obj = safe_json(model_path)
    baseline_obj = safe_json(baseline_path)

    model_vals = model_extractor(model_obj) if model_obj else {}
    if baseline_extractor:
        baseline_vals = baseline_extractor(baseline_obj) if baseline_obj else {}
    else:
        baseline_vals = model_extractor(baseline_obj) if baseline_obj else {}

    for metric, direction in direction_by_metric.items():
        mv = model_vals.get(metric)
        bv = baseline_vals.get(metric)
        status = "ok" if mv is not None else "missing"
        rows.append(
            MetricRow(
                task=task,
                metric=metric,
                direction=direction,
                model_value=mv,
                baseline_value=bv,
                delta=compute_delta(mv, bv, direction),
                model_artifact=Path(model_path).name if model_path else "",
                baseline_artifact=Path(baseline_path).name if baseline_path else "",
                status=status,
            )
        )
    return rows


def fmt(x: Optional[float], prec: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{prec}f}"


def write_markdown(path: Path, rows: List[MetricRow]):
    unique_tasks = sorted({r.task for r in rows})
    complete = sum(any(r.status == "ok" for r in rows if r.task == t) for t in unique_tasks)
    lines = [
        "# Foundation-Model Downstream Report",
        "",
        f"Coverage: {complete}/{len(unique_tasks)} tasks have at least one metric.",
        "",
        "| task | metric | dir | model | baseline | delta | model_artifact | baseline_artifact | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.task,
                    r.metric,
                    "↑" if r.direction == "higher" else "↓",
                    fmt(r.model_value),
                    fmt(r.baseline_value),
                    fmt(r.delta),
                    r.model_artifact or "n/a",
                    r.baseline_artifact or "n/a",
                    r.status,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: List[MetricRow]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "metric",
                "direction",
                "model_value",
                "baseline_value",
                "delta",
                "model_artifact",
                "baseline_artifact",
                "status",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def latex_escape(text: str) -> str:
    mapping = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    return "".join(mapping.get(ch, ch) for ch in text)


def fmt_tex(x: Optional[float], prec: int = 4) -> str:
    return "--" if x is None else f"{x:.{prec}f}"


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").replace("@", " @")


def task_label(task: str) -> str:
    return task.replace("_", " ")


def write_latex_full_table(path: Path, rows: List[MetricRow]):
    row_end = r"\\"
    lines = [
        "% Auto-generated by route_rangers.cli.generate_foundation_report",
        "% Requires: \\usepackage{booktabs,longtable}",
        "\\begin{longtable}{llccccc}",
        "\\caption{Unified downstream results across available repository evaluations. Positive $\\Delta$ indicates improvement for the specified metric direction.}\\label{tab:foundation_downstream_full}\\\\",
        "\\toprule",
        "Task & Metric & Dir & Model & Baseline & $\\Delta$ & Status " + row_end,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Task & Metric & Dir & Model & Baseline & $\\Delta$ & Status " + row_end,
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]

    prev_task: Optional[str] = None
    for r in rows:
        if prev_task is not None and r.task != prev_task:
            lines.append("\\midrule")
        prev_task = r.task
        dir_symbol = "$\\uparrow$" if r.direction == "higher" else "$\\downarrow$"
        row = "{} & {} & {} & {} & {} & {} & {}".format(
            latex_escape(task_label(r.task)),
            latex_escape(metric_label(r.metric)),
            dir_symbol,
            fmt_tex(r.model_value),
            fmt_tex(r.baseline_value),
            fmt_tex(r.delta),
            latex_escape(r.status),
        )
        lines.append(row + " " + row_end)

    lines.append("\\end{longtable}")
    path.write_text("\n".join(lines) + "\n")


def write_latex_task_summary(path: Path, rows: List[MetricRow]):
    tasks = sorted({r.task for r in rows})
    row_end = r"\\"
    lines = [
        "% Auto-generated by route_rangers.cli.generate_foundation_report",
        "% Requires: \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Task-level summary over all generated metrics. Wins/Losses/Ties are computed only where both model and baseline are available.}",
        "\\label{tab:foundation_downstream_task_summary}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Task & Metrics & Baseline & Win & Loss & Tie & Missing " + row_end,
        "\\midrule",
    ]

    for task in tasks:
        task_rows = [r for r in rows if r.task == task]
        total = len(task_rows)
        with_baseline = sum(
            r.model_value is not None and r.baseline_value is not None and r.delta is not None
            for r in task_rows
        )
        win = sum(r.delta is not None and r.delta > 1e-12 for r in task_rows)
        loss = sum(r.delta is not None and r.delta < -1e-12 for r in task_rows)
        tie = sum(r.delta is not None and abs(r.delta) <= 1e-12 for r in task_rows)
        missing = sum(r.model_value is None for r in task_rows)
        row = "{} & {} & {} & {} & {} & {} & {}".format(
            latex_escape(task_label(task)),
            total,
            with_baseline,
            win,
            loss,
            tie,
            missing,
        )
        lines.append(row + " " + row_end)

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[MetricRow] = []

    rows += make_rows(
        task="benchmark_core",
        direction_by_metric={
            "recon_acc_l0": "higher",
            "recon_acc_l1": "higher",
            "recon_acc_l2": "higher",
            "next_top1": "higher",
            "next_top5": "higher",
            "dest_top1": "higher",
            "dest_top5": "higher",
        },
        model_path=latest_match(args.cache_dir, "benchmark_*.json", exclude_substr=["baseline", "smoke", "%j"]),
        baseline_path=latest_match(args.cache_dir, "benchmark_*.json", include_substr="baseline"),
        model_extractor=extract_benchmark,
    )

    rows += make_rows(
        task="embedding_retrieval",
        direction_by_metric={"top1": "higher", "top5": "higher"},
        model_path=latest_match(args.cache_dir, "embedding_retrieval_*.json", exclude_substr=["baseline"]),
        baseline_path=latest_match(args.cache_dir, "embedding_retrieval_*.json", include_substr="baseline"),
        model_extractor=extract_embedding_retrieval,
    )

    rows += make_rows(
        task="change_detection",
        direction_by_metric={"auc": "higher", "margin": "higher"},
        model_path=latest_match(args.cache_dir, "change_detection_*.json", exclude_substr=["baseline", "quick"]),
        baseline_path=latest_match(args.cache_dir, "change_detection_*.json", include_substr="baseline"),
        model_extractor=extract_change_detection,
    )

    rows += make_rows(
        task="reverse_order",
        direction_by_metric={"delta_top1": "higher", "delta_nll": "lower"},
        model_path=latest_match(args.cache_dir, "reverse_order_*.json", exclude_substr=["baseline", "quick"]),
        baseline_path=latest_match(args.cache_dir, "reverse_order_*.json", include_substr="baseline"),
        model_extractor=extract_reverse_order,
    )

    rows += make_rows(
        task="invariance_suite",
        direction_by_metric={
            "prefix_1.00_dest_top1": "higher",
            "downsample_0.25_dest_top1": "higher",
        },
        model_path=latest_match(args.cache_dir, "invariance_*.json", exclude_substr=["baseline"]),
        baseline_path=latest_match(args.cache_dir, "invariance_*.json", include_substr="baseline"),
        model_extractor=extract_invariance,
    )

    rows += make_rows(
        task="length_sensitivity",
        direction_by_metric={"long_dest_top1": "higher", "gap_dest_top1": "higher"},
        model_path=latest_match(args.cache_dir, "length_sensitivity_*.json", exclude_substr=["smoke"]),
        baseline_path=None,
        model_extractor=extract_length,
    )

    rows += make_rows(
        task="travel_time",
        direction_by_metric={
            "mae_s@0.5": "lower",
            "rmse_s@0.5": "lower",
            "mape@0.5": "lower",
        },
        model_path=latest_match(args.cache_dir, "travel_time_*.json"),
        baseline_path=None,
        model_extractor=extract_travel_time,
    )

    rows += make_rows(
        task="anomaly_detection",
        direction_by_metric={"combined_auroc": "higher", "combined_pr_auc": "higher"},
        model_path=latest_match(args.cache_dir, "anomaly_detection_*.json"),
        baseline_path=None,
        model_extractor=extract_anomaly,
    )

    rows += make_rows(
        task="trip_classification",
        direction_by_metric={
            "speed_mode_top1": "higher",
            "duration_bucket_top1": "higher",
            "distance_bucket_top1": "higher",
        },
        model_path=latest_match(args.cache_dir, "trip_classification_*.json"),
        baseline_path=None,
        model_extractor=extract_trip_classification,
    )

    rows += make_rows(
        task="similarity_retrieval",
        direction_by_metric={
            "self_retrieval_top1": "higher",
            "self_retrieval_top10": "higher",
            "origin_mean_dist_km": "lower",
            "dest_mean_dist_km": "lower",
            "spatial_mean_dist_km": "lower",
        },
        model_path=latest_match(args.cache_dir, "similarity_retrieval_*.json"),
        baseline_path=None,
        model_extractor=extract_similarity,
    )

    rows += make_rows(
        task="macro_eval",
        direction_by_metric={
            "macro_kl": "lower",
            "macro_js": "lower",
            "macro_l1": "lower",
            "macro_top1": "higher",
        },
        model_path=latest_match(args.cache_dir, "macro_eval_*.json"),
        baseline_path=None,
        model_extractor=extract_macro_eval,
    )

    rows += make_rows(
        task="transfer_suite",
        direction_by_metric={
            "recovery_mae_m_mean": "lower",
            "prediction_mae_m_mean": "lower",
        },
        model_path=latest_match(args.cache_dir, "unitraj_transfer_suite_*.json"),
        baseline_path=None,
        model_extractor=extract_transfer_suite,
    )

    rows += make_rows(
        task="next_poi_eval",
        direction_by_metric={
            "next_poi_top1": "higher",
            "next_poi_top5": "higher",
            "next_poi_top10": "higher",
            "next_poi_ndcg10": "higher",
            "next_poi_mrr": "higher",
            "user_id_top1": "higher",
        },
        model_path=latest_match(args.cache_dir, "next_poi_eval_*.json"),
        baseline_path=None,
        model_extractor=extract_next_poi_eval,
    )

    rows += make_rows(
        task="cross_city_transfer",
        direction_by_metric={
            "cross_city_top1_mean": "higher",
            "cross_city_top5_mean": "higher",
            "cross_city_top10_mean": "higher",
            "cross_city_ndcg10_mean": "higher",
            "cross_city_mrr_mean": "higher",
            "cross_city_user_id_top1_mean": "higher",
        },
        model_path=latest_match(args.cache_dir, "cross_city_transfer_*.json"),
        baseline_path=None,
        model_extractor=extract_cross_city_transfer,
    )

    rows += make_rows(
        task="unitraj_regression",
        direction_by_metric={"recovery_mae_m": "lower", "prediction_mae_m": "lower"},
        model_path=latest_match(args.cache_dir, "unitraj_eval_regression_*.json", exclude_substr=["robust"]),
        baseline_path=latest_match(args.cache_dir, "unitraj_external_eval_*.json"),
        model_extractor=extract_unitraj_regression,
        baseline_extractor=extract_external_unitraj,
    )

    rows += make_rows(
        task="unitraj_regression_robust",
        direction_by_metric={"recovery_mae_m": "lower", "prediction_mae_m": "lower"},
        model_path=latest_match(args.cache_dir, "unitraj_eval_regression_robust_*.json"),
        baseline_path=None,
        model_extractor=extract_unitraj_regression,
    )

    rows += make_rows(
        task="unitraj_centroid",
        direction_by_metric={"recovery_mae_m": "lower", "prediction_mae_m": "lower"},
        model_path=latest_match(
            args.cache_dir,
            "unitraj_eval_*.json",
            exclude_substr=["regression", "robust", "excl", "hash", "l1", "l2", "hmt_"],
        ),
        baseline_path=latest_match(args.cache_dir, "unitraj_external_eval_*.json"),
        model_extractor=extract_unitraj_regression,
        baseline_extractor=extract_external_unitraj,
    )

    rows += make_rows(
        task="unitraj_centroid_robust",
        direction_by_metric={"recovery_mae_m": "lower", "prediction_mae_m": "lower"},
        model_path=latest_match(
            args.cache_dir,
            "unitraj_eval_robust_*.json",
            exclude_substr=["hmt_"],
        ),
        baseline_path=None,
        model_extractor=extract_unitraj_regression,
    )

    rows.sort(key=lambda r: (r.task, r.metric))

    md_path = output_dir / f"{args.filename_prefix}.md"
    csv_path = output_dir / f"{args.filename_prefix}.csv"
    json_path = output_dir / f"{args.filename_prefix}.json"
    tex_full_path = output_dir / f"{args.filename_prefix}_full_table.tex"
    tex_summary_path = output_dir / f"{args.filename_prefix}_task_summary.tex"

    write_markdown(md_path, rows)
    write_csv(csv_path, rows)
    write_latex_full_table(tex_full_path, rows)
    write_latex_task_summary(tex_summary_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "cache_dir": args.cache_dir,
                "rows": [asdict(r) for r in rows],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {tex_full_path}")
    print(f"wrote {tex_summary_path}")


if __name__ == "__main__":
    main()
