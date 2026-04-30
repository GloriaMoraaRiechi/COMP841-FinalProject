"""
Solve one user-specified routing problem with trained CNN checkpoints and
produce a side-by-side BEFORE / AFTER visualisation in the style of the paper
reference figure.

Usage
-----
From Python:

    from solve_and_visualize import solve_and_visualize
    solve_and_visualize(
        grid_size=8,
        nets=[[(0,0),(7,7)], [(0,7),(7,0)], [(1,3),(6,3)], [(3,0),(3,7)]],
        cleaner_ckpt='checkpoints/cleaner_best.npz',
        router_ckpt='checkpoints/router_rl.npz',   # or router_sl.npz
        method='full_pipeline',
        out_path='results/custom/solution.png',
    )

From the shell (requires the training has been run):

    python solve_and_visualize.py \
        --grid_size 8 \
        --nets_json '[[[0,0],[7,7]],[[0,7],[7,0]],[[1,3],[6,3]]]' \
        --cleaner_ckpt checkpoints/cleaner_best.npz \
        --router_ckpt checkpoints/router_rl.npz \
        --method full_pipeline \
        --out results/custom/solution.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from models import CleanerScoringCNN, RouterPolicyValueNet
from pipeline import run_method
from routing_env import RoutingInstance


Cell = Tuple[int, int]


def _infer_cleaner_shape(state: dict) -> Tuple[int, int]:
    """Return (width, depth) from a cleaner state dict."""
    width = int(state["backbone.block0.conv.W"].shape[0])
    depth = 0
    while f"backbone.block{depth}.conv.W" in state:
        depth += 1
    return width, depth


def _infer_router_shape(state: dict) -> Tuple[int, int]:
    """Return (width, depth) from a router state dict."""
    width = int(state["backbone.block0.conv.W"].shape[0])
    depth = 0
    while f"backbone.block{depth}.conv.W" in state:
        depth += 1
    return width, depth


def load_cleaner(ckpt_path: str, in_channels: int, width: int = 0, depth: int = 0) -> CleanerScoringCNN:
    state = dict(np.load(ckpt_path))
    # If user passed 0, auto-detect from checkpoint.
    iw, idp = _infer_cleaner_shape(state)
    if width <= 0:
        width = iw
    if depth <= 0:
        depth = idp
    model = CleanerScoringCNN(in_channels=in_channels, width=width, depth=depth)
    model.load_state_dict(state)
    model.eval()
    return model


def load_router(ckpt_path: str, in_channels: int, width: int = 0, depth: int = 0) -> RouterPolicyValueNet:
    state = dict(np.load(ckpt_path))
    iw, idp = _infer_router_shape(state)
    if width <= 0:
        width = iw
    if depth <= 0:
        depth = idp
    model = RouterPolicyValueNet(in_channels=in_channels, width=width, depth=depth)
    model.load_state_dict(state)
    model.eval()
    return model


def validate_instance(grid_size: int, nets: Sequence) -> List[Tuple[Cell, Cell]]:
    """Coerce nets into tuple form and validate."""
    out: List[Tuple[Cell, Cell]] = []
    seen = set()
    for idx, item in enumerate(nets):
        src = tuple(int(v) for v in item[0])
        sink = tuple(int(v) for v in item[1])
        for name, cell in (("src", src), ("sink", sink)):
            r, c = cell
            if not (0 <= r < grid_size and 0 <= c < grid_size):
                raise ValueError(f"Net {idx} {name} cell {cell} is outside the {grid_size}x{grid_size} grid.")
            if cell in seen:
                raise ValueError(f"Pin cell {cell} is duplicated across nets.")
            seen.add(cell)
        if src == sink:
            raise ValueError(f"Net {idx} has the same source and sink: {src}.")
        out.append((src, sink))
    return out


def _pretty_plot_side(ax, grid_size: int, nets, paths, overlap_count: int, wire: int, title: str, colors) -> None:
    ax.set_title(title, fontsize=13)
    # Light grid
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color="#dddddd", lw=1)
        ax.axvline(i - 0.5, color="#dddddd", lw=1)
    ax.set_xlim(-0.7, grid_size - 0.3)
    ax.set_ylim(grid_size - 0.3, -0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    # Draw overlap highlight patches using the canonical strict-overlap rule
    # (so any overlap the METRIC counts is also shown pink in the figure).
    from routing_env import RoutingInstance, strict_overlap_map
    instance = RoutingInstance(
        grid_size=grid_size,
        nets=tuple((tuple(s), tuple(t)) for s, t in nets),
    )
    ov_map = strict_overlap_map(instance, paths)
    for r in range(grid_size):
        for c in range(grid_size):
            if ov_map[r, c] > 0:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            facecolor="#f7b7b7", edgecolor=None, alpha=0.75, zorder=1))

    # Draw paths (jitter very slightly so overlapping segments stay visible)
    for i, path in enumerate(paths):
        if not path:
            continue
        xs = [c for r, c in path]
        ys = [r for r, c in path]
        ax.plot(xs, ys, color=colors[i % len(colors)], lw=3, solid_capstyle="round", solid_joinstyle="round", zorder=2)

    # Draw pins on top
    for i, (src, sink) in enumerate(nets):
        ax.scatter([src[1]], [src[0]], marker="s", s=180, color=colors[i % len(colors)], edgecolor="black", lw=1, zorder=4)
        ax.scatter([sink[1]], [sink[0]], marker="o", s=180, color=colors[i % len(colors)], edgecolor="black", lw=1, zorder=4)

    ax.text(
        0.5, -0.12,
        f"overlap = {overlap_count}    wirelength = {wire}",
        transform=ax.transAxes,
        ha="center", va="top", fontsize=10,
    )


def _legend_handles(nets, colors):
    handles = []
    for i, _ in enumerate(nets):
        handles.append(mlines.Line2D([], [], color=colors[i % len(colors)], lw=3, label=f"net {i}"))
    handles.append(mlines.Line2D([], [], color="grey", lw=0, marker="s", markersize=10,
                                  markerfacecolor="grey", markeredgecolor="black", label="source"))
    handles.append(mlines.Line2D([], [], color="grey", lw=0, marker="o", markersize=10,
                                  markerfacecolor="grey", markeredgecolor="black", label="sink"))
    return handles


def solve_and_visualize(
    grid_size: int,
    nets: Sequence[Tuple[Cell, Cell]],
    cleaner: CleanerScoringCNN,
    router: RouterPolicyValueNet,
    method: str = "full_pipeline",
    rounds: int = 6,
    beam_width: int = 6,
    out_path: str = "solution.png",
    title_extra: str = "",
    rng_seed: int = 7,
) -> Dict:
    """Run the chosen pipeline on one instance and save BEFORE/AFTER figure."""
    nets_clean = validate_instance(grid_size, nets)
    instance = RoutingInstance(grid_size=grid_size, nets=tuple(nets_clean))

    record = run_method(instance, method, cleaner, router,
                         rounds=rounds, beam_width=beam_width, rng=random.Random(rng_seed))

    # Figure
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
               "#17becf", "#bcbd22", "#e377c2", "#8c564b", "#7f7f7f"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _pretty_plot_side(
        axes[0], grid_size, instance.nets, record["paths_before"],
        record["before_overlap"], record["before_wirelength"],
        "Before", colors,
    )
    _pretty_plot_side(
        axes[1], grid_size, instance.nets, record["paths_after"],
        record["after_overlap"], record["after_wirelength"],
        "After", colors,
    )
    handles = _legend_handles(instance.nets, colors)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
                frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.03))
    sup = f"Method: {method}"
    if title_extra:
        sup = sup + "  |  " + title_extra
    fig.suptitle(sup, fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return record


def _parse_nets_json(s: str) -> List[Tuple[Cell, Cell]]:
    raw = json.loads(s)
    nets = []
    for item in raw:
        src = tuple(int(v) for v in item[0])
        sink = tuple(int(v) for v in item[1])
        nets.append((src, sink))
    return nets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid_size", type=int, required=True)
    p.add_argument("--nets_json", type=str, required=True)
    p.add_argument("--cleaner_ckpt", type=str, default="checkpoints/cleaner_best.npz")
    p.add_argument("--router_ckpt", type=str, default="checkpoints/router_rl.npz")
    p.add_argument("--cleaner_width", type=int, default=0, help="0 = auto-detect from checkpoint.")
    p.add_argument("--cleaner_depth", type=int, default=0, help="0 = auto-detect from checkpoint.")
    p.add_argument("--router_width", type=int, default=0, help="0 = auto-detect from checkpoint.")
    p.add_argument("--router_depth", type=int, default=0, help="0 = auto-detect from checkpoint.")
    p.add_argument("--method", type=str, default="full_pipeline")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--beam_width", type=int, default=6)
    p.add_argument("--out", type=str, default="results/custom/solution.png")
    p.add_argument("--save_json", action="store_true")
    args = p.parse_args()

    cleaner = load_cleaner(args.cleaner_ckpt, in_channels=12, width=args.cleaner_width, depth=args.cleaner_depth)
    router = load_router(args.router_ckpt, in_channels=9, width=args.router_width, depth=args.router_depth)

    nets = _parse_nets_json(args.nets_json)
    record = solve_and_visualize(
        grid_size=args.grid_size,
        nets=nets,
        cleaner=cleaner,
        router=router,
        method=args.method,
        rounds=args.rounds,
        beam_width=args.beam_width,
        out_path=args.out,
    )
    print(f"Before overlap : {record['before_overlap']}  wire : {record['before_wirelength']}")
    print(f"After  overlap : {record['after_overlap']}  wire : {record['after_wirelength']}")
    print(f"Zero-overlap success: {bool(record['success'])}")
    print(f"Chosen nets: {record['chosen_nets']}")
    print(f"Figure saved to {args.out}")

    if args.save_json:
        out_json = Path(args.out).with_suffix(".json")
        def _p(paths):
            return [[[int(r), int(c)] for r, c in p] if p else None for p in paths]
        payload = dict(
            grid_size=args.grid_size,
            nets=[[list(s), list(t)] for s, t in nets],
            method=args.method,
            before_overlap=record["before_overlap"],
            after_overlap=record["after_overlap"],
            before_wirelength=record["before_wirelength"],
            after_wirelength=record["after_wirelength"],
            success=record["success"],
            chosen_nets=record["chosen_nets"],
            paths_before=_p(record["paths_before"]),
            paths_after=_p(record["paths_after"]),
        )
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"JSON saved to {out_json}")


if __name__ == "__main__":
    main()
