"""
Routing environment for the physical-design routing problem.

A routing instance is a square grid with N nets. Each net has a source pin and
a sink pin. A path is a list of 4-connected grid cells from source to sink. A
solution is one path per net. We care about:

    - completion      : all nets routed, no strict overlap
    - overlap         : how many cells (outside pin cells) are used by >1 net
    - wirelength      : total cells used (sum of (len(path)-1) for each routed net)

The A* router connects one pair (src, sink) respecting a `blocked` set.

The teacher for supervised cleaner/router training is an oracle procedure that
tries every candidate net for removal, reroutes it against everyone else, and
scores the result lexicographically by (overlap, wirelength, path_len).
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    import networkx as nx  # used only for stronger candidate enumeration
except Exception:  # pragma: no cover
    nx = None


Cell = Tuple[int, int]
PathType = List[Cell]

ACTION_TO_DELTA: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
DELTA_TO_ACTION: Dict[Tuple[int, int], int] = {v: k for k, v in ACTION_TO_DELTA.items()}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


# ---------------------------------------------------------------------------
# Instance + samplers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingInstance:
    grid_size: int
    nets: Tuple[Tuple[Cell, Cell], ...]

    @property
    def num_nets(self) -> int:
        return len(self.nets)

    @property
    def pin_cells(self) -> Set[Cell]:
        return {cell for src, sink in self.nets for cell in (src, sink)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(grid_size: int, cell: Cell) -> bool:
    r, c = cell
    return 0 <= r < grid_size and 0 <= c < grid_size


def iter_neighbors(grid_size: int, cell: Cell) -> Iterable[Cell]:
    r, c = cell
    for dr, dc in ACTION_TO_DELTA.values():
        nxt = (r + dr, c + dc)
        if in_bounds(grid_size, nxt):
            yield nxt


def action_from_step(src: Cell, dst: Cell) -> int:
    delta = (dst[0] - src[0], dst[1] - src[1])
    if delta not in DELTA_TO_ACTION:
        raise ValueError(f"Invalid step {src} -> {dst}")
    return DELTA_TO_ACTION[delta]


def sample_routing_instance(
    grid_size: int,
    num_nets: int,
    min_manhattan: int = 2,
    rng: Optional[np.random.Generator] = None,
    max_tries: int = 5000,
) -> RoutingInstance:
    rng = rng or np.random.default_rng()
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]

    for _ in range(max_tries):
        idx = rng.choice(len(all_cells), size=2 * num_nets, replace=False)
        cells = [all_cells[i] for i in idx]
        nets = []
        ok = True
        for i in range(num_nets):
            src = cells[2 * i]
            sink = cells[2 * i + 1]
            if manhattan(src, sink) < min_manhattan:
                ok = False
                break
            nets.append((src, sink))
        if ok:
            return RoutingInstance(grid_size=grid_size, nets=tuple(nets))

    raise RuntimeError(
        f"Failed to sample a valid instance after {max_tries} tries."
    )


def sample_instance_batch(
    count: int,
    grid_sizes: Sequence[int],
    num_nets_choices: Sequence[int],
    min_manhattan: int = 2,
    seed: int = 42,
) -> List[RoutingInstance]:
    rng = np.random.default_rng(seed)
    instances: List[RoutingInstance] = []
    gs_list = list(grid_sizes)
    nn_list = list(num_nets_choices)
    for _ in range(count):
        grid_size = int(rng.choice(gs_list))
        num_nets = int(rng.choice(nn_list))
        instances.append(
            sample_routing_instance(
                grid_size=grid_size,
                num_nets=num_nets,
                min_manhattan=min_manhattan,
                rng=rng,
            )
        )
    return instances


# ---------------------------------------------------------------------------
# A* router
# ---------------------------------------------------------------------------

def astar(
    grid_size: int,
    source: Cell,
    sink: Cell,
    blocked: Optional[Set[Cell]] = None,
    cost_map: Optional[np.ndarray] = None,
) -> Optional[PathType]:
    blocked = blocked or set()
    if source in blocked or sink in blocked:
        return None

    open_heap: List[Tuple[float, float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (float(manhattan(source, sink)), 0.0, counter, source))
    came_from: Dict[Cell, Optional[Cell]] = {source: None}
    g_score: Dict[Cell, float] = {source: 0.0}

    while open_heap:
        _, g_curr, _, current = heapq.heappop(open_heap)
        if current == sink:
            path: PathType = []
            node: Optional[Cell] = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            return path[::-1]

        if g_curr > g_score.get(current, float("inf")):
            continue

        for nxt in iter_neighbors(grid_size, current):
            if nxt in blocked and nxt != sink:
                continue
            cell_cost = 0.0
            if cost_map is not None:
                cell_cost = float(cost_map[nxt[0], nxt[1]])
            new_g = g_curr + 1.0 + cell_cost
            if new_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_g
                f = new_g + manhattan(nxt, sink)
                came_from[nxt] = current
                counter += 1
                heapq.heappush(open_heap, (f, new_g, counter, nxt))
    return None


def route_all_independent(instance: RoutingInstance) -> List[Optional[PathType]]:
    """Each net is routed independently. Wires are free to overlap each other
    (that is the whole point — that is what the cleaner+router exists to fix)
    but no net is allowed to route through ANOTHER net's source or sink pin."""
    paths: List[Optional[PathType]] = []
    for i, (src, sink) in enumerate(instance.nets):
        blocked: Set[Cell] = set()
        for j, (s, t) in enumerate(instance.nets):
            if j == i:
                continue
            blocked.add(s); blocked.add(t)
        paths.append(astar(instance.grid_size, src, sink, blocked=blocked))
    return paths


# ---------------------------------------------------------------------------
# Overlap / wirelength metrics
# ---------------------------------------------------------------------------

def occupancy_count(paths: Sequence[Optional[Sequence[Cell]]], grid_size: int) -> np.ndarray:
    occ = np.zeros((grid_size, grid_size), dtype=np.float32)
    for path in paths:
        if not path:
            continue
        for r, c in path:
            occ[r, c] += 1.0
    return occ


def strict_overlap_map(instance: RoutingInstance, paths) -> np.ndarray:
    """Strict electrical overlap.

    Rule: each grid cell may be used by at most one net. A pin cell (source or
    sink of some net X) is LEGITIMATELY touched only by net X. If any OTHER
    net's path passes through net X's pin cell, that is an overlap (the wires
    are shorted together).

    Returns a (H, W) float32 map with 1.0 on overlap cells.
    """
    gs = instance.grid_size
    overlap = np.zeros((gs, gs), dtype=np.float32)

    # Map pin cell -> owning net index (pin cells are unique by construction).
    pin_owner: Dict[Cell, int] = {}
    for i, (src, sink) in enumerate(instance.nets):
        pin_owner[src] = i
        pin_owner[sink] = i

    # For each cell collect the set of net indices touching it.
    cell_users: Dict[Cell, Set[int]] = {}
    for idx, path in enumerate(paths):
        if not path:
            continue
        for cell in path:
            cell_users.setdefault(cell, set()).add(idx)

    for cell, users in cell_users.items():
        if cell in pin_owner:
            owner = pin_owner[cell]
            # Overlap if any non-owner net touches this pin cell.
            foreign = users - {owner}
            if foreign:
                overlap[cell[0], cell[1]] = 1.0
        else:
            # Non-pin cell: overlap if more than one net uses it.
            if len(users) > 1:
                overlap[cell[0], cell[1]] = 1.0

    return overlap


def strict_overlap_count(instance: RoutingInstance, paths) -> int:
    return int(strict_overlap_map(instance, paths).sum())


def literal_overlap_map(instance: RoutingInstance, paths) -> np.ndarray:
    occ = occupancy_count(paths, instance.grid_size)
    return (occ > 1).astype(np.float32)


def literal_overlap_count(instance: RoutingInstance, paths) -> int:
    return int(literal_overlap_map(instance, paths).sum())


def total_wirelength(paths) -> int:
    total = 0
    for path in paths:
        if path and len(path) >= 2:
            total += len(path) - 1
    return total


# ---------------------------------------------------------------------------
# Reroute helpers for the cleaner / router
# ---------------------------------------------------------------------------

def remove_net(paths, net_idx: int) -> List[Optional[PathType]]:
    new_paths = [list(p) if p else None for p in paths]
    new_paths[net_idx] = None
    return new_paths


def blocked_cells_for_target(instance: RoutingInstance, partial_paths, net_idx: int) -> Set[Cell]:
    """Cells the target net MUST NOT use when being rerouted.

    Includes:
      - every cell on any OTHER net's path
      - every OTHER net's source and sink pin

    The target net's own source and sink are NOT in the set (it needs to enter
    them). Other nets' pins are blocked because routing through them creates
    an electrical short between nets.
    """
    src, sink = instance.nets[net_idx]
    blocked: Set[Cell] = set()
    for idx, path in enumerate(partial_paths):
        if idx == net_idx:
            continue
        if path:
            blocked.update(path)
        # Even if the other net has no path right now, its pins are off-limits.
        other_src, other_sink = instance.nets[idx]
        blocked.add(other_src)
        blocked.add(other_sink)
    blocked.discard(src)
    blocked.discard(sink)
    return blocked


def route_one_net_with_blocking(
    instance: RoutingInstance,
    partial_paths,
    net_idx: int,
    cost_map: Optional[np.ndarray] = None,
) -> Optional[PathType]:
    src, sink = instance.nets[net_idx]
    blocked = blocked_cells_for_target(instance, partial_paths, net_idx)
    return astar(instance.grid_size, src, sink, blocked=blocked, cost_map=cost_map)


def evaluate_net_reroute(instance: RoutingInstance, paths, net_idx: int) -> Dict[str, object]:
    """Oracle counterfactual: remove net, reroute it once, return scored result.

    Ranking priority (lower is better on each):
        1) strict overlap after
        2) total wirelength after
        3) new path length
        4) net index (deterministic)
    """
    before_overlap = strict_overlap_count(instance, paths)
    before_literal = literal_overlap_count(instance, paths)
    before_wire = total_wirelength(paths)
    old_path = paths[net_idx]
    old_len = max(0, len(old_path) - 1) if old_path else 0

    partial_paths = remove_net(paths, net_idx)
    new_path = route_one_net_with_blocking(instance, partial_paths, net_idx)

    if new_path is None:
        rank_key = (max(before_overlap, 1), before_wire + 999, old_len + 999, net_idx)
        score = -float(100.0 * rank_key[0] + 0.05 * rank_key[1] + 0.005 * rank_key[2] + 1e-4 * rank_key[3])
        return dict(
            net_idx=net_idx, success=False,
            before_overlap=before_overlap, before_literal_overlap=before_literal, before_total_wire=before_wire,
            after_overlap=max(before_overlap, 1), after_literal_overlap=max(before_literal, 1), after_total_wire=before_wire + 999,
            old_len=old_len, new_len=None,
            rank_key=rank_key, score=score,
            partial_paths=partial_paths, new_paths=None, new_path=None,
        )

    new_paths = [list(p) if p else None for p in partial_paths]
    new_paths[net_idx] = new_path
    after_overlap = strict_overlap_count(instance, new_paths)
    after_literal = literal_overlap_count(instance, new_paths)
    after_wire = total_wirelength(new_paths)
    new_len = max(0, len(new_path) - 1)

    rank_key = (after_overlap, after_wire, new_len, net_idx)
    score = -float(100.0 * rank_key[0] + 0.05 * rank_key[1] + 0.005 * rank_key[2] + 1e-4 * rank_key[3])

    return dict(
        net_idx=net_idx, success=True,
        before_overlap=before_overlap, before_literal_overlap=before_literal, before_total_wire=before_wire,
        after_overlap=after_overlap, after_literal_overlap=after_literal, after_total_wire=after_wire,
        old_len=old_len, new_len=new_len,
        rank_key=rank_key, score=score,
        partial_paths=partial_paths, new_paths=new_paths, new_path=new_path,
    )


def oracle_cleaner_scores(instance: RoutingInstance, paths):
    return [evaluate_net_reroute(instance, paths, i) for i in range(instance.num_nets)]


def best_oracle_candidate(instance: RoutingInstance, paths):
    return max(oracle_cleaner_scores(instance, paths), key=lambda x: float(x["score"]))


# ---------------------------------------------------------------------------
# Backtracking exact solver (used as a strong non-ML baseline)
# ---------------------------------------------------------------------------

def enumerate_candidate_paths(
    instance: RoutingInstance,
    net_idx: int,
    occupied: Optional[Set[Cell]] = None,
    max_paths: int = 12,
    extra_margin: int = 8,
) -> List[PathType]:
    occupied = set(occupied or set())
    src, sink = instance.nets[net_idx]
    blocked = set(occupied)
    blocked.discard(src)
    blocked.discard(sink)

    if nx is not None:
        g = nx.grid_2d_graph(instance.grid_size, instance.grid_size)
        if blocked:
            g.remove_nodes_from([cell for cell in blocked if cell in g])
        if src not in g or sink not in g:
            return []
        try:
            paths_iter = nx.shortest_simple_paths(g, src, sink)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        found: List[PathType] = []
        best_len: Optional[int] = None
        cutoff: Optional[int] = None
        try:
            for path in paths_iter:
                path = list(path)
                plen = len(path) - 1
                if best_len is None:
                    best_len = plen
                    cutoff = best_len + max(0, int(extra_margin))
                if cutoff is not None and plen > cutoff:
                    break
                found.append(path)
                if len(found) >= max_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return found

    # Fallback — just A* once
    p = astar(instance.grid_size, src, sink, blocked=blocked)
    return [p] if p else []


def solve_instance_exact(
    instance: RoutingInstance,
    max_paths_per_net: int = 12,
    extra_margin: int = 8,
) -> Optional[List[PathType]]:
    routes: List[Optional[PathType]] = [None] * instance.num_nets
    cache: Dict[Tuple[int, Tuple[Cell, ...]], List[PathType]] = {}

    def candidates_for(net_idx: int, occupied: Set[Cell]) -> List[PathType]:
        key = (net_idx, tuple(sorted(occupied)))
        if key not in cache:
            cache[key] = enumerate_candidate_paths(
                instance, net_idx, occupied=occupied,
                max_paths=max_paths_per_net, extra_margin=extra_margin,
            )
        return cache[key]

    def recurse(remaining: Set[int], occupied: Set[Cell]) -> bool:
        if not remaining:
            return True
        option_bank = []
        for net_idx in remaining:
            cands = candidates_for(net_idx, occupied)
            if not cands:
                return False
            src, sink = instance.nets[net_idx]
            option_bank.append((len(cands), -manhattan(src, sink), net_idx, cands))
        option_bank.sort(key=lambda item: (item[0], item[1], item[2]))
        _, _, chosen_net, chosen_cands = option_bank[0]
        for path in chosen_cands:
            routes[chosen_net] = path
            if recurse(remaining - {chosen_net}, occupied | set(path)):
                return True
            routes[chosen_net] = None
        return False

    ok = recurse(set(range(instance.num_nets)), set())
    if not ok:
        return None
    return [list(p) if p is not None else None for p in routes]


def distance_map(grid_size: int, sink: Cell) -> np.ndarray:
    dist = np.zeros((grid_size, grid_size), dtype=np.float32)
    denom = max(1, 2 * (grid_size - 1))
    for r in range(grid_size):
        for c in range(grid_size):
            dist[r, c] = manhattan((r, c), sink) / float(denom)
    return dist
