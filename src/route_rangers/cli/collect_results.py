#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect latest experiment results into markdown."
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument(
        "--foundation_doc", type=str, default="docs/foundation_evals.md"
    )
    parser.add_argument("--paper_doc", type=str, default="docs/neurips_paper_draft.md")
    return parser.parse_args()


def latest_files(pattern: str) -> List[str]:
    files = glob.glob(pattern)
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=lambda p: os.path.getmtime(p))
    return files


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def summarize_benchmarks(cache_dir: str) -> Tuple[str, List[Dict[str, str]]]:
    rows = []
    for path in latest_files(os.path.join(cache_dir, "benchmark_*.json")):
        name = Path(path).stem.replace("benchmark_", "")
        if any(tag in name for tag in ("smoke", "%j", "new_metrics")):
            continue
        obj = _safe_read_json(path)
        if not obj or "splits" not in obj:
            continue
        split = obj["splits"].get("random") or {}
        recon = split.get("reconstruction", {})
        next_loc = split.get("next_location_probe", {}).get("test", {})
        dest = split.get("destination_probe", {}).get("test", {})
        rows.append(
            {
                "name": name,
                "recon_l1": f"{recon.get('recon_acc_l1', 0.0):.3f}",
                "next_top1": f"{next_loc.get('top1', 0.0):.3f}",
                "dest_top1": f"{dest.get('top1', 0.0):.3f}",
            }
        )
    if not rows:
        return "No benchmark results yet.", []
    header = "| run | recon@l1 | next@top1 | dest@top1 |\n|---|---|---|---|"
    lines = [header]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['recon_l1']} | {r['next_top1']} | {r['dest_top1']} |"
        )
    return "\n".join(lines), rows


def _pick_split(obj: dict) -> Tuple[Optional[str], Optional[dict]]:
    splits = obj.get("splits") or {}
    if not splits:
        return None, None
    for key in ("random", "temporal", "all"):
        if key in splits:
            return key, splits[key]
    key = next(iter(splits.keys()))
    return key, splits.get(key)


def _pick_metric(task: dict) -> dict:
    if not task:
        return {}
    for key in ("test", "val", "train"):
        if key in task:
            return task[key]
    return task


def summarize_unitraj_eval(
    cache_dir: str, pattern: str, label: str, exclude_substr: Optional[str] = None
) -> str:
    files = latest_files(os.path.join(cache_dir, pattern))
    if exclude_substr:
        files = [f for f in files if exclude_substr not in Path(f).name]
    if not files:
        return f"No {label} results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return f"No {label} results yet."
    split_name, split = _pick_split(obj)
    lines = [f"latest: `{Path(path).name}`"]
    if split_name:
        lines.append(f"- split: {split_name}")
    if split:
        for task in ("recovery", "prediction"):
            t = _pick_metric(split.get(task, {}))
            if t:
                coverage = ""
                if "n_total" in t and t["n_total"]:
                    coverage = f", coverage={t.get('n', 0) / max(1, t['n_total']):.3f}"
                lines.append(
                    f"- {task}: mae_m={t.get('mae_m', 0.0):.1f}, "
                    f"rmse_m={t.get('rmse_m', 0.0):.1f}, "
                    f"n={t.get('n', 0)}{coverage}"
                )
    return "\n".join(lines)


def summarize_unitraj_external(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "unitraj_external_eval_*.json"))
    if not files:
        return "No external UniTraj results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No external UniTraj results yet."
    metrics = obj.get("metrics", {})
    lines = [f"latest: `{Path(path).name}`"]
    for task in ("recovery", "prediction"):
        t = metrics.get(task, {})
        if t:
            lines.append(
                f"- {task}: mae_m={t.get('mae_m', 0.0):.1f}, "
                f"rmse_m={t.get('rmse_m', 0.0):.1f}, "
                f"n={t.get('n', 0)}"
            )
    return "\n".join(lines)


def summarize_macro_eval(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "macro_eval_*.json"))
    if not files:
        return "No macro-head results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No macro-head results yet."
    metrics = obj.get("metrics", {})
    lines = [f"latest: `{Path(path).name}`"]
    for key in ("macro_kl", "macro_js", "macro_l1", "macro_top1", "n"):
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    return "\n".join(lines)


def summarize_commuting_zone(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "commuting_zone_probe_*.json"))
    if not files:
        return "No commuting-zone results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No commuting-zone results yet."
    metrics = obj.get("metrics", {})
    lines = [f"latest: `{Path(path).name}`"]
    for key in ("cz_val_acc", "cz_test_acc", "num_classes", "n"):
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    return "\n".join(lines)


def summarize_data_efficiency(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "unitraj_data_efficiency_*.json"))
    if not files:
        return "No data-efficiency results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No data-efficiency results yet."
    fractions = obj.get("fractions", {})
    if not fractions:
        return "No data-efficiency results yet."
    lines = [f"latest: `{Path(path).name}`"]
    fracs = sorted(fractions.keys(), key=lambda x: float(x))
    rec_parts = []
    pred_parts = []
    for frac in fracs:
        res = fractions.get(frac, {})
        split_name, split = _pick_split(res)
        if not split:
            continue
        rec = _pick_metric(split.get("recovery", {}))
        pred = _pick_metric(split.get("prediction", {}))
        if rec:
            rec_parts.append(f"{frac}:{rec.get('mae_m', 0.0):.1f}")
        if pred:
            pred_parts.append(f"{frac}:{pred.get('mae_m', 0.0):.1f}")
    if rec_parts:
        lines.append("- recovery mae_m: " + ", ".join(rec_parts))
    if pred_parts:
        lines.append("- prediction mae_m: " + ", ".join(pred_parts))
    return "\n".join(lines)


def summarize_transfer(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "unitraj_transfer_suite_*.json"))
    if not files:
        return "No transfer-suite results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No transfer-suite results yet."
    datasets = obj.get("datasets", {})
    if not datasets:
        return "No transfer-suite results yet."
    lines = [f"latest: `{Path(path).name}`"]
    for ds, res in datasets.items():
        split_name, split = _pick_split(res)
        if not split:
            continue
        rec = _pick_metric(split.get("recovery", {}))
        pred = _pick_metric(split.get("prediction", {}))
        ds_name = Path(ds).name
        parts = [f"{ds_name}"]
        if split_name:
            parts.append(f"split={split_name}")
        if rec:
            parts.append(f"recovery_mae_m={rec.get('mae_m', 0.0):.1f}")
        if pred:
            parts.append(f"pred_mae_m={pred.get('mae_m', 0.0):.1f}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def summarize_length(cache_dir: str) -> str:
    # Prefer dest-masked runs (avoid trivial destination copying).
    files = latest_files(os.path.join(cache_dir, "length_sensitivity_*destmask*.json"))
    if not files:
        files = latest_files(os.path.join(cache_dir, "length_sensitivity_*.json"))
    if not files:
        return "No length-sensitivity results yet."

    # Show a compact table over the most recent runs so we can compare ablations.
    n_show = 8
    recent = files[-min(n_show, len(files)) :]
    recent = list(reversed(recent))  # newest first

    rows = []
    latest_obj = None
    latest_name = Path(recent[0]).name if recent else Path(files[-1]).name
    latest_bins = None
    latest_strategy = None

    for path in recent:
        obj = _safe_read_json(path)
        if not obj:
            continue
        if latest_obj is None:
            latest_obj = obj
            latest_bins = obj.get("bins")
            latest_strategy = obj.get("bin_strategy")

        name = Path(path).stem.replace("length_sensitivity_", "")
        metrics = obj.get("metrics", {}) or {}
        gap = obj.get("length_sensitivity_gap", {}) or {}
        short = metrics.get("short", {}) or {}
        long = metrics.get("long", {}) or {}

        def _f(x):
            try:
                return float(x)
            except Exception:
                return 0.0

        if not gap and short and long:
            gap = {
                "recon_acc_l1": _f(long.get("recon_acc_l1"))
                - _f(short.get("recon_acc_l1")),
                "dest_top1": _f(long.get("dest_top1")) - _f(short.get("dest_top1")),
            }

        rows.append(
            {
                "run": name,
                "short_recon": _f(short.get("recon_acc_l1")),
                "long_recon": _f(long.get("recon_acc_l1")),
                "gap_recon": _f(gap.get("recon_acc_l1")),
                "short_dest": _f(short.get("dest_top1")),
                "long_dest": _f(long.get("dest_top1")),
                "gap_dest": _f(gap.get("dest_top1")),
                "n": int(obj.get("samples", 0)),
            }
        )

    if not rows:
        return "No length-sensitivity results yet."

    header = (
        "| run | recon@l1 short | recon@l1 long | gap"
        " | dest@top1 short | dest@top1 long | gap | n |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [f"latest: `{latest_name}`"]
    if latest_bins is not None:
        lines.append(f"bins: {latest_bins} ({latest_strategy})")
    lines.append("")
    lines.append(header)
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['short_recon']:.3f} | {r['long_recon']:.3f}"
            f" | {r['gap_recon']:.3f} | {r['short_dest']:.3f}"
            f" | {r['long_dest']:.3f} | {r['gap_dest']:.3f} | {r['n']} |"
        )
    return "\n".join(lines)


def summarize_invariance(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "invariance_*.json"))
    if not files:
        return "No invariance-suite results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No invariance-suite results yet."
    prefix = obj.get("prefix_destination", {})
    time_shift = obj.get("time_shift_destination", {})
    down = obj.get("downsample_destination", {})
    lines = [f"latest: `{Path(path).name}`"]
    if prefix:
        lines.append(
            "- prefix dest_top1: "
            + ", ".join(f"{k}:{v.get('dest_top1', 0.0):.3f}" for k, v in prefix.items())
        )
    if time_shift:
        lines.append(
            "- time-shift dest_top1: "
            + ", ".join(
                f"{k}:{v.get('dest_top1', 0.0):.3f}" for k, v in time_shift.items()
            )
        )
    if down:
        lines.append(
            "- downsample dest_top1: "
            + ", ".join(f"{k}:{v.get('dest_top1', 0.0):.3f}" for k, v in down.items())
        )
    return "\n".join(lines)


def summarize_simple(cache_dir: str, pattern: str, label: str, keys: List[str]) -> str:
    files = latest_files(os.path.join(cache_dir, pattern))
    if not files:
        return f"No {label} results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return f"No {label} results yet."
    name = Path(path).name
    tag = " (quick)" if "quick" in name else ""
    parts = [f"latest: `{name}`{tag}"]
    for k in keys:
        if k in obj:
            parts.append(f"- {k}: {obj[k]}")
    return "\n".join(parts)


def summarize_travel_time(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "travel_time_*.json"))
    if not files:
        return "No travel-time estimation results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No travel-time estimation results yet."
    lines = [f"latest: `{Path(path).name}`"]
    results = obj.get("results", {})
    if results:
        header = (
            "| prefix_ratio | MAE (s) | RMSE (s) | MAPE | R² |\n|---|---|---|---|---|"
        )
        lines.append(header)
        for ratio, metrics in sorted(results.items(), key=lambda x: float(x[0])):
            lines.append(
                f"| {ratio} | {metrics.get('mae_seconds', 0):.1f} | "
                f"{metrics.get('rmse_seconds', 0):.1f} | "
                f"{metrics.get('mape', 0):.3f} | "
                f"{metrics.get('r2', 0):.3f} |"
            )
    return "\n".join(lines)


def summarize_anomaly_detection(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "anomaly_detection_*.json"))
    if not files:
        return "No anomaly detection results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No anomaly detection results yet."
    lines = [f"latest: `{Path(path).name}`"]
    results = obj.get("results", {})
    if results:
        header = "| anomaly_type | AUROC | PR-AUC |\n|---|---|---|"
        lines.append(header)
        for atype in ("noise", "reverse", "swap", "detour", "combined"):
            m = results.get(atype, {})
            if m:
                lines.append(
                    f"| {atype} | {m.get('auroc', 0):.3f} | {m.get('pr_auc', 0):.3f} |"
                )
    return "\n".join(lines)


def summarize_trip_classification(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "trip_classification_*.json"))
    if not files:
        return "No trip classification results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No trip classification results yet."
    lines = [f"latest: `{Path(path).name}`"]
    results = obj.get("results", {})
    if results:
        header = "| task | accuracy | macro_f1 | n_classes |\n|---|---|---|---|"
        lines.append(header)
        for task_name in ("speed_mode", "duration_bucket", "distance_bucket"):
            m = results.get(task_name, {})
            if m:
                lines.append(
                    f"| {task_name} | {m.get('accuracy', 0):.3f} | "
                    f"{m.get('macro_f1', 0):.3f} | {m.get('n_classes', 0)} |"
                )
    return "\n".join(lines)


def summarize_similarity_retrieval(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "similarity_retrieval_*.json"))
    if not files:
        return "No similarity retrieval results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No similarity retrieval results yet."
    lines = [f"latest: `{Path(path).name}`"]
    results = obj.get("results", {})
    for section in ("geographic_knn", "self_retrieval"):
        m = results.get(section, {})
        if m:
            lines.append(
                f"- **{section}**: "
                + ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in m.items()
                )
            )
    return "\n".join(lines)


def summarize_unitraj_regression(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "unitraj_eval_regression_*.json"))
    if not files:
        return "No regression-based UniTraj results yet."
    # Exclude robust variant
    clean = [f for f in files if "robust" not in Path(f).name]
    robust = [f for f in files if "robust" in Path(f).name]

    parts = []
    for label, flist in [("clean", clean), ("robust", robust)]:
        if not flist:
            continue
        path = flist[-1]
        obj = _safe_read_json(path)
        if not obj:
            continue
        split_name, split = _pick_split(obj)
        parts.append(f"**{label}** (`{Path(path).name}`)")
        if split:
            for task in ("recovery", "prediction"):
                t = _pick_metric(split.get(task, {}))
                if t:
                    parts.append(
                        f"- {task}: mae_m={t.get('mae_m', 0.0):.1f}, "
                        f"rmse_m={t.get('rmse_m', 0.0):.1f}, "
                        f"n={t.get('n', 0)}"
                    )
    if not parts:
        return "No regression-based UniTraj results yet."
    return "\n".join(parts)


def summarize_proposal_baselines(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "proposal_baselines*.json"))
    if not files:
        return "No proposal baseline results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No proposal baseline results yet."
    split_name, split = _pick_split(obj)
    if not split:
        return f"latest: `{Path(path).name}` (no split payload)"

    mean_next = (
        split.get("mean_displacement", {})
        .get("next_location_regression_probe", {})
        .get("test", {})
    )
    mean_dest = (
        split.get("mean_displacement", {})
        .get("destination_regression_probe", {})
        .get("test", {})
    )
    rnn_next = (
        split.get("simple_rnn", {})
        .get("next_location_regression_probe", {})
        .get("test", {})
    )
    rnn_dest = (
        split.get("simple_rnn", {})
        .get("destination_regression_probe", {})
        .get("test", {})
    )
    lines = [f"latest: `{Path(path).name}`", f"- split: {split_name}"]
    lines.append("| baseline | next MAE (m) | next RMSE (m) | dest MAE (m) | dest RMSE (m) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| mean_displacement | {mean_next.get('mae_m', 0.0):.1f} | "
        f"{mean_next.get('rmse_m', 0.0):.1f} | {mean_dest.get('mae_m', 0.0):.1f} | "
        f"{mean_dest.get('rmse_m', 0.0):.1f} |"
    )
    lines.append(
        f"| simple_rnn | {rnn_next.get('mae_m', 0.0):.1f} | "
        f"{rnn_next.get('rmse_m', 0.0):.1f} | {rnn_dest.get('mae_m', 0.0):.1f} | "
        f"{rnn_dest.get('rmse_m', 0.0):.1f} |"
    )
    return "\n".join(lines)


def summarize_od_eval(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "od_eval*.json"))
    if not files:
        return "No OD-matrix evaluation results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No OD-matrix evaluation results yet."
    split_name, split = _pick_split(obj)
    if not split:
        return f"latest: `{Path(path).name}` (no split payload)"
    test = split.get("test", {})
    meta = split.get("meta", {})
    lines = [f"latest: `{Path(path).name}`", f"- split: {split_name}"]
    lines.append(
        f"- test: mae={test.get('mae', 0.0):.3f}, rmse={test.get('rmse', 0.0):.3f}, "
        f"mape={test.get('mape', 0.0):.3f}, rows={test.get('n_rows', 0)}"
    )
    lines.append(
        f"- meta: zones={meta.get('num_zones', 0)}, times={meta.get('num_times', 0)}"
    )
    return "\n".join(lines)


def summarize_length_uncertainty(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "length_uncertainty*.json"))
    if not files:
        return "No length-uncertainty results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No length-uncertainty results yet."
    metrics = obj.get("metrics", {})
    if not metrics:
        return "No length-uncertainty results yet."
    lines = [f"latest: `{Path(path).name}`"]
    lines.append("| bucket | next NLL | next entropy | dest NLL | dest entropy | dest ECE |")
    lines.append("|---|---|---|---|---|---|")
    for b in ("short", "medium", "long"):
        m = metrics.get(b, {})
        lines.append(
            f"| {b} | {m.get('next_step_nll', 0.0):.4f} | "
            f"{m.get('next_step_entropy', 0.0):.4f} | {m.get('dest_nll', 0.0):.4f} | "
            f"{m.get('dest_entropy', 0.0):.4f} | {m.get('dest_ece', 0.0):.4f} |"
        )
    return "\n".join(lines)


def summarize_embedding_length_probe(cache_dir: str) -> str:
    files = latest_files(os.path.join(cache_dir, "embedding_length_probe*.json"))
    if not files:
        return "No embedding-length probe results yet."
    path = files[-1]
    obj = _safe_read_json(path)
    if not obj:
        return "No embedding-length probe results yet."
    split_name, split = _pick_split(obj)
    if not split:
        return f"latest: `{Path(path).name}` (no split payload)"
    test = split.get("test", {})
    lines = [f"latest: `{Path(path).name}`", f"- split: {split_name}"]
    lines.append(
        f"- test: acc={test.get('accuracy', 0.0):.3f}, "
        f"macro_f1={test.get('macro_f1', 0.0):.3f}, n={test.get('n', 0)}"
    )
    return "\n".join(lines)


def replace_block(text: str, start: str, end: str, new_block: str) -> str:
    if start not in text or end not in text:
        return text
    pre = text.split(start)[0]
    post = text.split(end)[1]
    return f"{pre}{start}\n{new_block}\n{end}{post}"


def main():
    args = parse_args()
    cache_dir = args.cache_dir

    bench_md, _ = summarize_benchmarks(cache_dir)
    length_md = summarize_length(cache_dir)
    invariance_md = summarize_invariance(cache_dir)
    retrieval_md = summarize_simple(
        cache_dir,
        "embedding_retrieval_*.json",
        "embedding retrieval",
        ["top1", "top5", "samples"],
    )
    reverse_md = summarize_simple(
        cache_dir,
        "reverse_order_*.json",
        "reverse-order",
        ["original", "reversed", "delta"],
    )
    change_md = summarize_simple(
        cache_dir,
        "change_detection_*.json",
        "change detection",
        ["pos_mean_dist", "neg_mean_dist", "auc"],
    )
    unitraj_md = summarize_unitraj_eval(
        cache_dir, "unitraj_eval_*.json", "UniTraj-style eval", exclude_substr="robust"
    )
    unitraj_robust_md = summarize_unitraj_eval(
        cache_dir, "unitraj_eval_robust_*.json", "robust UniTraj eval"
    )
    unitraj_external_md = summarize_unitraj_external(cache_dir)
    data_eff_md = summarize_data_efficiency(cache_dir)
    transfer_md = summarize_transfer(cache_dir)
    macro_md = summarize_macro_eval(cache_dir)
    cz_md = summarize_commuting_zone(cache_dir)

    # New downstream tasks
    travel_time_md = summarize_travel_time(cache_dir)
    anomaly_md = summarize_anomaly_detection(cache_dir)
    trip_class_md = summarize_trip_classification(cache_dir)
    sim_retrieval_md = summarize_similarity_retrieval(cache_dir)
    unitraj_reg_md = summarize_unitraj_regression(cache_dir)
    baseline_md = summarize_proposal_baselines(cache_dir)
    od_md = summarize_od_eval(cache_dir)
    uncertainty_md = summarize_length_uncertainty(cache_dir)
    emb_len_probe_md = summarize_embedding_length_probe(cache_dir)

    summary = "\n\n".join(
        [
            "### Benchmarks (random split, test)",
            bench_md,
            "### Proposal Baselines",
            baseline_md,
            "### Length sensitivity",
            length_md,
            "### Length Uncertainty",
            uncertainty_md,
            "### Embedding Length Probe",
            emb_len_probe_md,
            "### OD Matrix Evaluation",
            od_md,
            "### Invariance suite",
            invariance_md,
            "### Embedding retrieval",
            retrieval_md,
            "### Reverse-order stress",
            reverse_md,
            "### Change detection",
            change_md,
            "### UniTraj-style eval (centroid)",
            unitraj_md,
            "### UniTraj-style eval (centroid, robust)",
            unitraj_robust_md,
            "### UniTraj-style eval (regression)",
            unitraj_reg_md,
            "### Data efficiency",
            data_eff_md,
            "### Transfer suite",
            transfer_md,
            "### Macro distribution head",
            macro_md,
            "### Commuting zone probe",
            cz_md,
            "### External UniTraj baseline",
            unitraj_external_md,
            "### Travel Time Estimation",
            travel_time_md,
            "### Anomaly Detection",
            anomaly_md,
            "### Trip Classification",
            trip_class_md,
            "### Similarity Retrieval",
            sim_retrieval_md,
        ]
    )

    # Update foundation_evals.md
    foundation_path = Path(args.foundation_doc)
    if foundation_path.exists():
        text = foundation_path.read_text()
        if "<!-- RESULTS:BEGIN -->" not in text:
            text += (
                "\n\n## Latest Results\n"
                "<!-- RESULTS:BEGIN -->\n(pending)\n<!-- RESULTS:END -->\n"
            )
        text = replace_block(
            text, "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->", summary
        )
        foundation_path.write_text(text)

    # Update paper draft
    paper_path = Path(args.paper_doc)
    if paper_path.exists():
        text = paper_path.read_text()
        if "<!-- RESULTS:BEGIN -->" not in text:
            text += (
                "\n\n## Results (Living Section)\n"
                "<!-- RESULTS:BEGIN -->\n(pending)\n<!-- RESULTS:END -->\n"
            )
        text = replace_block(
            text, "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->", summary
        )
        paper_path.write_text(text)

    print("updated results blocks")


if __name__ == "__main__":
    main()
