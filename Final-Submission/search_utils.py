"""
Inference-time search utilities.

At inference, the router policy can be used in three ways:

    greedy_policy_rollout : pick argmax valid action at each step
    policy_beam_search    : keep top-B partial paths scored by log-prob +
                            a mild manhattan heuristic

Beam search is slower but substantially more robust when a single greedy pick
would get stuck against blockers — it lets the router consider alternate sub-
routes whose first step might not be the highest-probability local pick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np

from dataset_generation import build_router_features
from nn import softmax
from routing_env import (
    ACTION_TO_DELTA,
    Cell,
    RoutingInstance,
    blocked_cells_for_target,
    manhattan,
)


@dataclass
class SearchResult:
    success: bool
    path: Optional[List[Cell]]
    score: float
    steps: int


def _next_cells(instance: RoutingInstance, cell: Cell):
    for action, (dr, dc) in ACTION_TO_DELTA.items():
        nxt = (cell[0] + dr, cell[1] + dc)
        if 0 <= nxt[0] < instance.grid_size and 0 <= nxt[1] < instance.grid_size:
            yield action, nxt


def greedy_policy_rollout(
    model,
    instance: RoutingInstance,
    partial_paths,
    net_idx: int,
    max_steps: Optional[int] = None,
) -> SearchResult:
    src, sink = instance.nets[net_idx]
    max_steps = max_steps or (instance.grid_size * instance.grid_size * 2)
    blocked = blocked_cells_for_target(instance, partial_paths, net_idx)

    prefix = [src]
    visited: Set[Cell] = {src}
    total_log = 0.0

    for step in range(max_steps):
        if prefix[-1] == sink:
            return SearchResult(True, prefix, total_log, step)
        x = build_router_features(instance, partial_paths, net_idx, prefix)
        logits, _ = model.forward(x[None])
        probs = softmax(logits[0:1], axis=-1)[0]

        candidates = []
        for a, nxt in _next_cells(instance, prefix[-1]):
            if nxt in blocked and nxt != sink:
                continue
            if nxt in visited and nxt != sink:
                continue
            candidates.append((a, nxt, float(probs[a])))

        if not candidates:
            return SearchResult(False, None, total_log, step)

        a, nxt, p = max(candidates, key=lambda t: t[2])
        total_log += math.log(max(p, 1e-8))
        prefix.append(nxt)
        visited.add(nxt)

    if prefix[-1] == sink:
        return SearchResult(True, prefix, total_log, max_steps)
    return SearchResult(False, None, total_log, max_steps)


def policy_beam_search(
    model,
    instance: RoutingInstance,
    partial_paths,
    net_idx: int,
    beam_width: int = 6,
    max_steps: Optional[int] = None,
    heuristic_coef: float = 0.08,
) -> SearchResult:
    src, sink = instance.nets[net_idx]
    max_steps = max_steps or (instance.grid_size * instance.grid_size * 2)
    blocked = blocked_cells_for_target(instance, partial_paths, net_idx)

    # (path, log-prob score, visited_set, done)
    beams: List[Tuple[List[Cell], float, Set[Cell]]] = [([src], 0.0, {src})]
    finished: List[Tuple[List[Cell], float]] = []

    for step in range(max_steps):
        expanded: List[Tuple[List[Cell], float, Set[Cell]]] = []
        # Batch forward passes for efficiency
        live = [(p, s, v) for p, s, v in beams if p[-1] != sink]
        if not live:
            break
        xs = np.stack([build_router_features(instance, partial_paths, net_idx, p) for p, _, _ in live]).astype(np.float32)
        logits, _ = model.forward(xs)
        probs = softmax(logits, axis=-1)
        for (p, s, v), pi in zip(live, probs):
            head = p[-1]
            for a, nxt in _next_cells(instance, head):
                if nxt in blocked and nxt != sink:
                    continue
                if nxt in v and nxt != sink:
                    continue
                step_score = math.log(max(float(pi[a]), 1e-8))
                heur = -heuristic_coef * manhattan(nxt, sink)
                new_score = s + step_score + heur
                new_v = set(v); new_v.add(nxt)
                new_path = p + [nxt]
                if nxt == sink:
                    finished.append((new_path, new_score))
                else:
                    expanded.append((new_path, new_score, new_v))

        if not expanded:
            break
        expanded.sort(key=lambda t: t[1], reverse=True)
        beams = expanded[:beam_width]

    if finished:
        finished.sort(key=lambda t: t[1], reverse=True)
        best_path, best_score = finished[0]
        return SearchResult(True, best_path, best_score, len(best_path) - 1)
    return SearchResult(False, None, float("-inf"), max_steps)


def policy_beam_search_topk(
    model,
    instance: RoutingInstance,
    partial_paths,
    net_idx: int,
    beam_width: int = 8,
    top_k: int = 6,
    max_steps: Optional[int] = None,
    heuristic_coef: float = 0.08,
) -> List[SearchResult]:
    """Same as policy_beam_search but returns up to `top_k` distinct finished
    paths ordered by log-prob. The pipeline uses this to pick the candidate
    that minimises GLOBAL overlap, not just the single highest-prob path."""
    src, sink = instance.nets[net_idx]
    max_steps = max_steps or (instance.grid_size * instance.grid_size * 2)
    blocked = blocked_cells_for_target(instance, partial_paths, net_idx)

    beams: List[Tuple[List[Cell], float, Set[Cell]]] = [([src], 0.0, {src})]
    finished: List[Tuple[List[Cell], float]] = []

    for _ in range(max_steps):
        expanded: List[Tuple[List[Cell], float, Set[Cell]]] = []
        live = [(p, s, v) for p, s, v in beams if p[-1] != sink]
        if not live:
            break
        xs = np.stack([build_router_features(instance, partial_paths, net_idx, p) for p, _, _ in live]).astype(np.float32)
        logits, _ = model.forward(xs)
        probs = softmax(logits, axis=-1)
        for (p, s, v), pi in zip(live, probs):
            head = p[-1]
            for a, nxt in _next_cells(instance, head):
                if nxt in blocked and nxt != sink:
                    continue
                if nxt in v and nxt != sink:
                    continue
                step_score = math.log(max(float(pi[a]), 1e-8))
                heur = -heuristic_coef * manhattan(nxt, sink)
                new_score = s + step_score + heur
                new_v = set(v); new_v.add(nxt)
                new_path = p + [nxt]
                if nxt == sink:
                    finished.append((new_path, new_score))
                else:
                    expanded.append((new_path, new_score, new_v))
        if not expanded:
            break
        expanded.sort(key=lambda t: t[1], reverse=True)
        beams = expanded[:beam_width]

    if not finished:
        return []
    # Deduplicate by path tuple, keep best-score occurrence.
    by_path = {}
    for p, s in finished:
        tup = tuple(p)
        if tup not in by_path or s > by_path[tup]:
            by_path[tup] = s
    unique = [(list(k), v) for k, v in by_path.items()]
    unique.sort(key=lambda t: t[1], reverse=True)
    return [SearchResult(True, path, score, len(path) - 1) for path, score in unique[:top_k]]
