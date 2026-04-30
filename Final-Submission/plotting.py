"""
Plotting helpers: training curves and benchmark bar charts.

Uses matplotlib's "Agg" backend so nothing tries to open a window.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_cleaner_history(history_path: Path, out_dir: Path) -> None:
    hist = json.loads(Path(history_path).read_text())
    epochs = [e["epoch"] for e in hist["epochs"]]
    loss = [e["loss"] for e in hist["epochs"]]
    train_top1 = [e["train_top1"] for e in hist["epochs"]]
    val_top1 = [e["val_top1"] for e in hist["epochs"]]

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, loss, marker="o", color="tab:red")
    axes[0].set_title("Cleaner: training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_top1, marker="o", label="train", color="tab:blue")
    axes[1].plot(epochs, val_top1, marker="s", label="val", color="tab:orange")
    axes[1].set_title("Cleaner: top-1 accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "cleaner_training.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_router_sl_history(history_path: Path, out_dir: Path) -> None:
    hist = json.loads(Path(history_path).read_text())
    epochs = [e["epoch"] for e in hist["epochs"]]
    loss = [e["loss"] for e in hist["epochs"]]
    train_acc = [e["train_step_acc"] for e in hist["epochs"]]
    val_acc = [e["val_step_acc"] for e in hist["epochs"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, loss, marker="o", color="tab:red")
    axes[0].set_title("Router: training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", label="train", color="tab:blue")
    axes[1].plot(epochs, val_acc, marker="s", label="val", color="tab:orange")
    axes[1].set_title("Router: step accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "router_sl_training.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_router_rl_history(history_path: Path, out_dir: Path) -> None:
    hist = json.loads(Path(history_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Targeted-repair / short adaptation format
    if hist.get("epochs"):
        epochs = hist["epochs"]
        xs = [e.get("epoch", i + 1) for i, e in enumerate(epochs)]
        loss = [e.get("loss", float("nan")) for e in epochs]
        comp = [e.get("completion", e.get("val_completion", float("nan"))) for e in epochs]
        ov = [e.get("overlap", e.get("val_overlap", float("nan"))) for e in epochs]
        base = hist.get("baseline_validation", {})
        base_comp = base.get("completion", None)
        base_ov = base.get("overlap", None)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(xs, loss, marker="o")
        axes[0].set_title("Router post-stage: training loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(xs, comp, marker="o")
        if base_comp is not None:
            axes[1].axhline(base_comp, linestyle="--", label=f"baseline = {base_comp:.3f}")
            axes[1].legend(loc="lower right")
        axes[1].set_title("Router post-stage: completion")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("completion")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(alpha=0.3)

        axes[2].plot(xs, ov, marker="o")
        if base_ov is not None:
            axes[2].axhline(base_ov, linestyle="--", label=f"baseline = {base_ov:.2f}")
            axes[2].legend(loc="upper right")
        axes[2].set_title("Router post-stage: overlap")
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("avg overlap")
        axes[2].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "router_rl_training.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    if not hist.get("episodes"):
        return
    eps = [e["episode"] for e in hist["episodes"]]

    # Detect format: conjoint RL has window_overlap & window_wirelength;
    # pipeline-RL has window_completion + window_overlap_reduction;
    # single-net RL has window_success.
    is_reward_aligned = ("val_completion" in hist["episodes"][0] and "val_overlap" in hist["episodes"][0])
    is_conjoint = ("window_overlap" in hist["episodes"][0] and "window_wirelength" in hist["episodes"][0])
    is_pipeline = ("window_completion" in hist["episodes"][0] and "window_overlap_reduction" in hist["episodes"][0])
    if is_reward_aligned:
        comp = [e["val_completion"] for e in hist["episodes"]]
        ov = [e["val_overlap"] for e in hist["episodes"]]
        rew = [e.get("window_reward", 0.0) for e in hist["episodes"]]
        acc = [e.get("window_accept_rate", 0.0) for e in hist["episodes"]]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].plot(eps, comp, marker="o", color="tab:green")
        axes[0].set_title("Router RL: validation completion")
        axes[0].set_xlabel("episode"); axes[0].set_ylabel("completion")
        axes[0].set_ylim(0.0, 1.05); axes[0].grid(alpha=0.3)
        axes[1].plot(eps, ov, marker="o", color="tab:red")
        axes[1].set_title("Router RL: validation overlap")
        axes[1].set_xlabel("episode"); axes[1].set_ylabel("avg overlap")
        axes[1].grid(alpha=0.3)
        axes[2].plot(eps, rew, marker="o", color="tab:blue")
        axes[2].set_title("Router RL: rollout reward")
        axes[2].set_xlabel("episode"); axes[2].set_ylabel("reward")
        axes[2].grid(alpha=0.3)
        axes[3].plot(eps, acc, marker="o", color="tab:purple")
        axes[3].set_title("Router RL: accepted update rate")
        axes[3].set_xlabel("episode"); axes[3].set_ylabel("fraction")
        axes[3].set_ylim(0.0, 1.05); axes[3].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "router_rl_training.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    elif is_conjoint:
        comp = [e["window_completion"] for e in hist["episodes"]]
        ov = [e["window_overlap"] for e in hist["episodes"]]
        wire = [e["window_wirelength"] for e in hist["episodes"]]
        reward = [e["window_total_reward"] for e in hist["episodes"]]
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        axes[0].plot(eps, comp, marker="o", color="tab:green")
        axes[0].set_title("Router RL: completion rate")
        axes[0].set_xlabel("episode"); axes[0].set_ylabel("fraction")
        axes[0].set_ylim(0.0, 1.05); axes[0].grid(alpha=0.3)
        axes[1].plot(eps, ov, marker="o", color="tab:red")
        axes[1].set_title("Router RL: avg overlap")
        axes[1].set_xlabel("episode"); axes[1].set_ylabel("strict overlap")
        axes[1].grid(alpha=0.3)
        axes[2].plot(eps, wire, marker="o", color="tab:blue")
        axes[2].set_title("Router RL: avg wirelength")
        axes[2].set_xlabel("episode"); axes[2].set_ylabel("wire")
        axes[2].grid(alpha=0.3)
        axes[3].plot(eps, reward, marker="o", color="tab:purple")
        axes[3].set_title("Router RL: episode reward")
        axes[3].set_xlabel("episode"); axes[3].set_ylabel("sum of step rewards")
        axes[3].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "router_rl_training.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    elif is_pipeline:
        comp = [e["window_completion"] for e in hist["episodes"]]
        ov_red = [e["window_overlap_reduction"] for e in hist["episodes"]]
        reward = [e["window_total_reward"] for e in hist["episodes"]]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(eps, comp, marker="o", color="tab:green")
        axes[0].set_title("Pipeline-RL: full-instance completion")
        axes[0].set_xlabel("episode"); axes[0].set_ylabel("fraction with overlap=0")
        axes[0].set_ylim(0.0, 1.05); axes[0].grid(alpha=0.3)
        axes[1].plot(eps, ov_red, marker="o", color="tab:blue")
        axes[1].set_title("Pipeline-RL: overlap reduction (units)")
        axes[1].set_xlabel("episode"); axes[1].set_ylabel("avg (initial overlap - final overlap)")
        axes[1].grid(alpha=0.3)
        axes[2].plot(eps, reward, marker="o", color="tab:purple")
        axes[2].set_title("Pipeline-RL: total episode reward")
        axes[2].set_xlabel("episode"); axes[2].set_ylabel("sum of step rewards")
        axes[2].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "router_rl_training.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    else:
        succ = [e["window_success"] for e in hist["episodes"]]
        reward = [e["window_reward"] for e in hist["episodes"]]
        greedy = [e.get("greedy_success", float("nan")) for e in hist["episodes"]]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(eps, succ, marker="o", label="sampled episode success", color="tab:blue")
        axes[0].plot(eps, greedy, marker="s", label="greedy success", color="tab:orange")
        axes[0].set_title("Single-net RL success")
        axes[0].set_xlabel("episode"); axes[0].set_ylabel("fraction")
        axes[0].set_ylim(0.0, 1.02); axes[0].grid(alpha=0.3); axes[0].legend(loc="lower right")
        axes[1].plot(eps, reward, marker="o", color="tab:green")
        axes[1].set_title("Single-net RL reward (windowed)")
        axes[1].set_xlabel("episode"); axes[1].set_ylabel("total reward per episode")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "router_rl_training.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_benchmark_bars(summary_path: Path, out_dir: Path) -> None:
    data = json.loads(Path(summary_path).read_text())
    methods = list(data["methods"].keys())
    labels = [data["methods"][m]["label"] for m in methods]
    comp = [data["methods"][m]["completion_rate"] for m in methods]
    ov_after = [data["methods"][m]["avg_overlap_after"] for m in methods]
    wire_after = [data["methods"][m]["avg_wirelength_after"] for m in methods]
    ov_before = data["methods"][methods[0]]["avg_overlap_before"]
    wire_before = data["methods"][methods[0]]["avg_wirelength_before"]

    # Color per method kind: baseline gray, classical blue, ML green.
    kind_colors = {
        "initial":                          "#9e9e9e",
        "pathfinder":                       "#1f77b4",
        "learned_cleaner_learned_router":   "#2ca02c",
    }
    colors = [kind_colors.get(m, "#4c72b0") for m in methods]

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(methods)), 4.5))
    x = np.arange(len(methods))
    bars = ax.bar(x, comp, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Completion rate (no strict overlap)")
    ax.set_title("Pipeline completion rate across methods")
    ax.grid(alpha=0.3, axis="y")
    for b, v in zip(bars, comp):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_completion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(methods)), 4.5))
    bars = ax.bar(x, ov_after, color=colors)
    ax.axhline(ov_before, color="grey", linestyle="--", label=f"avg overlap BEFORE = {ov_before:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average strict overlap after method")
    ax.set_title("Avg strict overlap  (lower is better)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    for b, v in zip(bars, ov_after):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_overlap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, 2 * len(methods)), 4.5))
    bars = ax.bar(x, wire_after, color=colors)
    ax.axhline(wire_before, color="grey", linestyle="--", label=f"avg wirelength BEFORE = {wire_before:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Average wirelength after method")
    ax.set_title("Avg total wirelength")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    for b, v in zip(bars, wire_after):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_wirelength.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Ablation table plot, if present ----
    if "ablation" in data and data["ablation"]:
        ablation = data["ablation"]
        labels = [r["label"] for r in ablation]
        comp = [r["completion_rate"] for r in ablation]
        ov_after = [r["avg_overlap_after"] for r in ablation]
        n = len(labels)

        # Color last bar (full system) green, all others gray
        colors = ["#9e9e9e"] * (n - 1) + ["#2ca02c"]

        fig, axes = plt.subplots(1, 2, figsize=(max(10, 2.0 * n), 5))

        bars = axes[0].bar(np.arange(n), comp, color=colors)
        axes[0].set_xticks(np.arange(n))
        axes[0].set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Completion rate")
        axes[0].set_title("Ablation: completion rate")
        axes[0].grid(alpha=0.3, axis="y")
        for b, v in zip(bars, comp):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                          ha="center", va="bottom", fontsize=10, fontweight="bold")

        bars = axes[1].bar(np.arange(n), ov_after, color=colors)
        axes[1].set_xticks(np.arange(n))
        axes[1].set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
        axes[1].set_ylabel("Average overlap after")
        axes[1].set_title("Ablation: avg residual overlap (lower is better)")
        axes[1].grid(alpha=0.3, axis="y")
        for b, v in zip(bars, ov_after):
            axes[1].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                          ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.tight_layout()
        fig.savefig(out_dir / "ablation.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def plot_dataset_stats(meta: Dict, splits: Dict, out_dir: Path) -> None:
    """Two plots: grid-size distribution and net-count distribution per split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Per-split counts
    split_names = ["train", "val", "test"]
    cleaner_counts = [splits[s]["cleaner_items"] for s in split_names]
    router_counts = [splits[s]["router_items"] for s in split_names]
    instance_counts = [splits[s]["instances"] for s in split_names]
    x = np.arange(len(split_names))
    width = 0.25
    axes[0].bar(x - width, instance_counts, width=width, label="instances", color="tab:blue")
    axes[0].bar(x, cleaner_counts, width=width, label="cleaner samples", color="tab:orange")
    axes[0].bar(x + width, router_counts, width=width, label="router samples", color="tab:green")
    axes[0].set_xticks(x); axes[0].set_xticklabels(split_names)
    axes[0].set_ylabel("count")
    axes[0].set_title("Dataset sample counts per split")
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis="y")
    for i, (ic, cc, rc) in enumerate(zip(instance_counts, cleaner_counts, router_counts)):
        axes[0].text(i - width, ic, str(ic), ha="center", va="bottom", fontsize=8)
        axes[0].text(i, cc, str(cc), ha="center", va="bottom", fontsize=8)
        axes[0].text(i + width, rc, str(rc), ha="center", va="bottom", fontsize=8)

    # Grid-size distribution
    grid_sizes = meta.get("grid_sizes", [])
    nets_choices = meta.get("num_nets_choices", [])
    axes[1].axis("off")
    txt = [
        "Dataset meta:",
        f"  total instances      : {meta.get('num_instances', '-')}",
        f"  grid sizes           : {grid_sizes}",
        f"  num nets choices     : {nets_choices}",
        f"  min manhattan        : {meta.get('min_manhattan', '-')}",
        f"  cleaner channels     : {meta.get('cleaner_channels', '-')}",
        f"  router  channels     : {meta.get('router_channels', '-')}",
        "",
        f"Train : {splits['train']['instances']:>5} inst  |  "
        f"{splits['train']['cleaner_items']:>6} cleaner  |  {splits['train']['router_items']:>6} router",
        f"Val   : {splits['val']['instances']:>5} inst  |  "
        f"{splits['val']['cleaner_items']:>6} cleaner  |  {splits['val']['router_items']:>6} router",
        f"Test  : {splits['test']['instances']:>5} inst  |  "
        f"{splits['test']['cleaner_items']:>6} cleaner  |  {splits['test']['router_items']:>6} router",
    ]
    axes[1].text(0.02, 0.98, "\n".join(txt), va="top", family="monospace", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_dir / "dataset_stats.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
