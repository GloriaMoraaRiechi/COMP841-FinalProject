"""
Dataset generation for the Cleaner and Router models.

Cleaner feature tensor (12 channels, C x H x W, values in [0, 1]):
    0  candidate_path_before   : 1 where the candidate net's *current* path lives
    1  other_paths_before      : normalised occupancy (clipped to [0,1]) of OTHER nets
    2  candidate_pins          : 1 on the candidate net's src/sink
    3  other_pins              : 1 on every OTHER net's src/sink
    4  all_paths_before        : normalised occupancy of all nets
    5  overlap_before_all      : 1 on cells where >1 net overlaps (strict)
    6  candidate_overlap_before: overlap cells that intersect the candidate
    7  candidate_path_after    : 1 on the counterfactual rerouted path
    8  all_paths_after         : normalised occupancy after the reroute
    9  overlap_after           : 1 on overlap cells after the reroute
   10  overlap_cleared         : overlap_before - overlap_after (in [0,1])
   11  success_flag            : constant 1 if reroute succeeded else 0

Router feature tensor (9 channels, C x H x W):
    0  blocked                 : 1 on cells occupied by OTHER nets
    1  other_occ_norm          : normalised occupancy of other nets
    2  routed_body             : 1 on the prefix already laid by the router (minus head)
    3  head_mask               : 1 on the router head
    4  sink_mask               : 1 on the sink
    5  src_mask                : 1 on the src
    6  visited_mask            : 1 on every visited cell (prefix)
    7  distance_map            : manhattan distance to sink, normalised to [0,1]
    8  valid_next              : 1 on cells that are legal next-steps (in-bounds, not blocked)

These are grid-size independent — the same model runs on 5x5, 8x8, 10x10, 12x12.

For a single routing instance we generate:
    - 1 cleaner *group*  of `num_nets` candidate items (one per net)
    - 1 router  *episode* of (len(best_new_path) - 1) step items

Over 20,000 instances with a sensible nets distribution this produces roughly
100,000 cleaner samples and 200,000+ router samples.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from routing_env import (
    ACTION_TO_DELTA,
    RoutingInstance,
    best_oracle_candidate,
    distance_map,
    evaluate_net_reroute,
    occupancy_count,
    oracle_cleaner_scores,
    route_all_independent,
    sample_instance_batch,
    strict_overlap_map,
    total_wirelength,
)

CLEANER_CHANNELS = 12
ROUTER_CHANNELS = 9


def _cell_mask(grid_size: int, cells) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in cells:
        mask[r, c] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Cleaner features
# ---------------------------------------------------------------------------

def build_cleaner_features(
    instance: RoutingInstance,
    paths,
    net_idx: int,
    candidate: Optional[Dict] = None,
) -> np.ndarray:
    grid_size = instance.grid_size
    if candidate is None:
        candidate = evaluate_net_reroute(instance, paths, net_idx)

    candidate_before = _cell_mask(grid_size, paths[net_idx] or [])
    other_occ = occupancy_count([p for i, p in enumerate(paths) if i != net_idx], grid_size)
    all_occ = occupancy_count(paths, grid_size)
    overlap_before = strict_overlap_map(instance, paths)

    src, sink = instance.nets[net_idx]
    cand_pins = _cell_mask(grid_size, [src, sink])
    other_pin_cells = []
    for i, (s, t) in enumerate(instance.nets):
        if i != net_idx:
            other_pin_cells.extend([s, t])
    other_pins = _cell_mask(grid_size, other_pin_cells)

    cand_overlap_before = candidate_before * overlap_before

    new_path = candidate.get("new_path")
    new_paths = candidate.get("new_paths")
    candidate_after = _cell_mask(grid_size, new_path or [])
    if new_paths is None:
        all_occ_after = np.zeros_like(all_occ)
        overlap_after = np.ones_like(overlap_before) * float(candidate.get("after_overlap", 0) > 0)
    else:
        all_occ_after = occupancy_count(new_paths, grid_size)
        overlap_after = strict_overlap_map(instance, new_paths)

    overlap_cleared = np.clip(overlap_before - overlap_after, 0.0, 1.0)
    success_flag = np.ones((grid_size, grid_size), dtype=np.float32) * float(bool(candidate.get("success", False)))

    feat = np.stack(
        [
            candidate_before,
            np.clip(other_occ, 0.0, 4.0) / 4.0,
            cand_pins,
            other_pins,
            np.clip(all_occ, 0.0, 6.0) / 6.0,
            overlap_before,
            cand_overlap_before,
            candidate_after,
            np.clip(all_occ_after, 0.0, 6.0) / 6.0,
            overlap_after,
            overlap_cleared,
            success_flag,
        ],
        axis=0,
    ).astype(np.float32)

    assert feat.shape[0] == CLEANER_CHANNELS
    return feat


# ---------------------------------------------------------------------------
# Router features
# ---------------------------------------------------------------------------

def build_router_features(instance: RoutingInstance, partial_paths, net_idx: int, prefix) -> np.ndarray:
    grid_size = instance.grid_size
    src, sink = instance.nets[net_idx]
    head = prefix[-1]

    other_paths = [p for i, p in enumerate(partial_paths) if i != net_idx]
    other_occ = occupancy_count(other_paths, grid_size)
    blocked = (other_occ > 0).astype(np.float32)

    routed_body = _cell_mask(grid_size, prefix[:-1])
    visited = _cell_mask(grid_size, prefix)
    head_mask = _cell_mask(grid_size, [head])
    sink_mask = _cell_mask(grid_size, [sink])
    src_mask = _cell_mask(grid_size, [src])
    dist = distance_map(grid_size, sink)

    valid_next = np.zeros((grid_size, grid_size), dtype=np.float32)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = head[0] + dr, head[1] + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and blocked[nr, nc] == 0 and (nr, nc) not in set(prefix[:-1]):
            valid_next[nr, nc] = 1.0

    feat = np.stack(
        [
            blocked,
            np.clip(other_occ, 0.0, 4.0) / 4.0,
            routed_body,
            head_mask,
            sink_mask,
            src_mask,
            visited,
            dist,
            valid_next,
        ],
        axis=0,
    ).astype(np.float32)

    assert feat.shape[0] == ROUTER_CHANNELS
    return feat


# ---------------------------------------------------------------------------
# Instance -> training items
# ---------------------------------------------------------------------------

def instance_to_cleaner_items(instance: RoutingInstance, group_id: int) -> Optional[List[Dict]]:
    paths = route_all_independent(instance)
    if any(p is None for p in paths):
        return None
    if strict_overlap_map(instance, paths).sum() <= 0:
        return None

    oracle = oracle_cleaner_scores(instance, paths)
    best_score = max(float(c["score"]) for c in oracle)
    items = []
    for cand in oracle:
        net_idx = int(cand["net_idx"])
        feat = build_cleaner_features(instance, paths, net_idx, candidate=cand)
        items.append(dict(
            group_id=group_id,
            grid_size=instance.grid_size,
            instance=instance,
            paths=paths,
            net_idx=net_idx,
            x=feat,
            target_score=float(cand["score"]),
            is_best=bool(abs(float(cand["score"]) - best_score) <= 1e-9),
            meta=dict(
                before_overlap=int(cand["before_overlap"]),
                after_overlap=int(cand["after_overlap"]),
                success=bool(cand["success"]),
            ),
        ))
    return items


def instance_to_router_items(instance: RoutingInstance, group_id: int) -> Optional[List[Dict]]:
    paths = route_all_independent(instance)
    if any(p is None for p in paths):
        return None
    if strict_overlap_map(instance, paths).sum() <= 0:
        return None

    best = best_oracle_candidate(instance, paths)
    if not bool(best["success"]) or best["new_path"] is None:
        return None

    net_idx = int(best["net_idx"])
    partial_paths = best["partial_paths"]
    path = best["new_path"]
    if len(path) < 2:
        return None

    items = []
    for step in range(len(path) - 1):
        prefix = path[: step + 1]
        delta = (path[step + 1][0] - path[step][0], path[step + 1][1] - path[step][1])
        action = int({(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}[delta])
        remaining = len(path) - step - 1
        items.append(dict(
            group_id=group_id,
            grid_size=instance.grid_size,
            instance=instance,
            partial_paths=partial_paths,
            net_idx=net_idx,
            prefix=list(prefix),
            x=build_router_features(instance, partial_paths, net_idx, prefix),
            action=action,
            value=-float(remaining),
            path=list(path),
        ))
    return items


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

def _split_list(items: List, ratios=(0.7, 0.15, 0.15)):
    n = len(items)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def generate_datasets(
    num_instances: int,
    grid_sizes: Sequence[int],
    num_nets_choices: Sequence[int],
    min_manhattan: int,
    seed: int,
    test_size: int = 200,
    test_seed: int = 12345,
) -> Dict:
    """Generate train/val/test splits.

    The test split is generated from a SEPARATE FIXED SEED (`test_seed`,
    default 12345) so that A* and PathFinder benchmark numbers are stable
    across training runs. If you change `--seed` on the training side, the
    train/val splits move but the test set stays the same.
    """
    # Fixed test set
    test_count = min(test_size, num_instances)
    test_instances = sample_instance_batch(
        count=test_count,
        grid_sizes=grid_sizes,
        num_nets_choices=num_nets_choices,
        min_manhattan=min_manhattan,
        seed=test_seed,
    )

    # Train+val from the user seed
    n_trainval = max(0, num_instances - test_count)
    if n_trainval == 0:
        # Edge case — generate at least train/val from the seed
        n_trainval = num_instances
    trainval_instances = sample_instance_batch(
        count=n_trainval,
        grid_sizes=grid_sizes,
        num_nets_choices=num_nets_choices,
        min_manhattan=min_manhattan,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    order = np.arange(len(trainval_instances))
    rng.shuffle(order)
    trainval_instances = [trainval_instances[i] for i in order]
    # Train = 80% of train+val, val = 20%
    n_train = int(len(trainval_instances) * 0.8)
    train_instances = trainval_instances[:n_train]
    val_instances = trainval_instances[n_train:]

    def build_split(split_instances: List[RoutingInstance], split_name: str):
        cleaner_items: List[Dict] = []
        router_items: List[Dict] = []
        gid = 0
        usable_cleaner = 0
        usable_router = 0
        for instance in split_instances:
            c = instance_to_cleaner_items(instance, gid)
            r = instance_to_router_items(instance, gid)
            if c:
                cleaner_items.extend(c)
                usable_cleaner += 1
            if r:
                router_items.extend(r)
                usable_router += 1
            gid += 1
        return dict(
            instances=split_instances,
            cleaner=cleaner_items,
            router=router_items,
            stats=dict(
                split_name=split_name,
                instances_total=len(split_instances),
                instances_with_overlap=usable_cleaner,
                instances_with_router_signal=usable_router,
            ),
        )

    return dict(
        train=build_split(train_instances, "train"),
        val=build_split(val_instances, "val"),
        test=build_split(test_instances, "test"),
        meta=dict(
            num_instances=num_instances,
            grid_sizes=list(grid_sizes),
            num_nets_choices=list(num_nets_choices),
            min_manhattan=min_manhattan,
            seed=seed,
            cleaner_channels=CLEANER_CHANNELS,
            router_channels=ROUTER_CHANNELS,
        ),
    )
