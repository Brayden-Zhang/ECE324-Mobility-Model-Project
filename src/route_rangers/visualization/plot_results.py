"""Generate the paper figures under docs/figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from route_rangers.config import PAPER_FIGURES_DIR, ensure_project_directories
from route_rangers.reporting import build_paper_metrics


COLORS = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "gray": "#7f7f7f",
}


def _configure_matplotlib() -> None:
    # Higher-level styling so the generated plots look submission-ready.
    plt.rcParams.update({"figure.facecolor": "white"})
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(PAPER_FIGURES_DIR / f"{stem}.png")
    fig.savefig(PAPER_FIGURES_DIR / f"{stem}.pdf")
    plt.close(fig)


def plot_length_and_lsg(metrics: dict) -> None:
    length = metrics.get("length", {})
    short = length.get("short_dest_top1") or 0.0
    long = length.get("long_dest_top1") or 0.0
    gap = length.get("gap_dest_top1") or (long - short)

    labels = ["Short", "Long"]
    values = [short, long]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(x, values, width=0.55, color=[COLORS["blue"], COLORS["orange"]], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dest-Local@Top1")
    ax.set_title("Length Sensitivity (Long - Short)")
    bins = length.get("bins")
    bins_line = f"Length bins: {bins[0]}, {bins[1]}" if bins else "Length bins from artifact"
    ax.text(
        0.03,
        0.12,
        f"{bins_line}\nLSG = {gap:.3f}",
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    _save(fig, "results_length_lsg")


def plot_robustness_suite(metrics: dict) -> None:
    inv = metrics.get("invariance", {})
    reverse_order = metrics.get("reverse_order", {})
    change = metrics.get("change_detection", {})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7))
    fig.subplots_adjust(wspace=0.35)

    prefix = inv.get("prefix_dest_top1", {})
    prefix_x = [0.25, 0.50, 0.75, 1.00]
    prefix_y = [prefix.get(f"{value:.2f}", 0.0) for value in prefix_x]
    axes[0].plot(prefix_x, prefix_y, marker="o", linewidth=2.5, color=COLORS["blue"])
    axes[0].set_title("Prefix Invariance")
    axes[0].set_xlabel("Observed ratio")
    axes[0].set_ylabel("Dest@Top1")
    axes[0].set_xticks(prefix_x)
    axes[0].set_xticklabels(["25%", "50%", "75%", "100%"])
    y_min, y_max = min(prefix_y), max(prefix_y)
    axes[0].set_ylim(max(0.0, y_min - 0.04), min(1.05, y_max + 0.04))

    labels = ["Original", "Reversed"]
    original = reverse_order.get("original", {})
    reversed_metrics = reverse_order.get("reversed", {})
    top1 = [original.get("top1", 0.0), reversed_metrics.get("top1", 0.0)]
    top5 = [original.get("top5", 0.0), reversed_metrics.get("top5", 0.0)]
    x = np.arange(len(labels))
    width = 0.34
    b1 = axes[1].bar(x - width / 2, top1, width, label="Top-1", color=COLORS["green"], zorder=3)
    b5 = axes[1].bar(x + width / 2, top5, width, label="Top-5", color=COLORS["blue"], zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Reverse-Order Stress")
    axes[1].legend(frameon=True)
    for bars in (b1, b5):
        for bar in bars:
            height = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    cd_labels = ["Pos dist", "Neg dist", "AUC"]
    cd_values = [
        change.get("pos_mean_dist", 0.0),
        change.get("neg_mean_dist", 0.0),
        change.get("auc", 0.0),
    ]
    bcd = axes[2].bar(cd_labels, cd_values, color=[COLORS["orange"], COLORS["gray"], COLORS["red"]], zorder=3)
    # auto-scale but keep AUC visible
    cd_max = max(cd_values) if cd_values else 1.0
    axes[2].set_ylim(0.0, min(1.2, cd_max + 0.1))
    axes[2].set_title("Change Detection")
    for bar in bcd:
        height = bar.get_height()
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, "results_robustness_suite")


def plot_meter_scale(metrics: dict) -> None:
    clean = metrics.get("unitraj_clean", {})
    recovery = clean.get("recovery", {})
    prediction = clean.get("prediction", {})
    hmt_mae = [recovery.get("hmt_mae_m", 0.0), prediction.get("hmt_mae_m", 0.0)]
    unitraj_mae = [
        recovery.get("baseline_mae_m", 0.0),
        prediction.get("baseline_mae_m", 0.0),
    ]

    hmt_rmse = [recovery.get("hmt_rmse_m", 0.0), prediction.get("hmt_rmse_m", 0.0)]
    unitraj_rmse = [
        recovery.get("baseline_rmse_m", 0.0),
        prediction.get("baseline_rmse_m", 0.0),
    ]
    tasks = ["Recovery", "Prediction"]
    x = np.arange(len(tasks))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
    axes[0].bar(x - width / 2, hmt_mae, width, label="HMT", color=COLORS["blue"], zorder=3)
    axes[0].bar(x + width / 2, unitraj_mae, width, label="UniTraj", color=COLORS["gray"], zorder=3)
    axes[0].set_title("MAE (m)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks)
    axes[0].set_ylabel("Meters")

    axes[1].bar(x - width / 2, hmt_rmse, width, label="HMT", color=COLORS["blue"], zorder=3)
    axes[1].bar(x + width / 2, unitraj_rmse, width, label="UniTraj", color=COLORS["gray"], zorder=3)
    axes[1].set_title("RMSE (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks)
    axes[1].legend(frameon=True)

    _save(fig, "results_meter_scale")


def main() -> None:
    ensure_project_directories()
    _configure_matplotlib()
    metrics = build_paper_metrics()
    plot_length_and_lsg(metrics)
    plot_robustness_suite(metrics)
    plot_meter_scale(metrics)
    print(f"Wrote paper figures to {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
