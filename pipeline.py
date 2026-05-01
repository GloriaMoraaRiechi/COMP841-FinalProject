"""
End-to-end routing pipelines and baselines.

Baseline (non-ML, widely used in physical-design routing):
    - A* per-net, independent              (classical greedy)
    - NegotiationBasedRouter (PathFinder)  (classical, iterative rip-up & reroute
                                            with a history-based cost map; this
                                            is the standard independent baseline
                                            for routing with overlap resolution)
    - Exact backtracking                   (small-grid solver for comparison)

ML pipelines:
    - learned_cleaner_learned_router       : CNN cleaner picks the net to rip
                                             up, CNN router reroutes it
    - hybrid_cleaner_learned_router        : cleaner score is used as a tie-break
                                             over the oracle rank key
    - learned_cleaner_astar_router         : ablation (only cleaner is ML)
    - astar_cleaner_learned_router         : ablation (only router is ML)
    - full_pipeline                        : ML first, PathFinder fallback,
                                             exact fallback (small grids)

Every method returns the same record shape so the evaluator can compare them
on: completion_rate (no overlap), avg overlap, avg wirelength.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from dataset_generation import build_cleaner_features
from routing_env import (
    RoutingInstance,
    astar,
    best_oracle_candidate,
    enumerate_candidate_paths,
    blocked_cells_for_target,
    evaluate_net_reroute,
    literal_overlap_count,
    remove_net,
    route_all_independent,
    solve_instance_exact,
    strict_overlap_count,
    strict_overlap_map,
    total_wirelength,
)
from search_utils import greedy_policy_rollout, policy_beam_search


# ---------------------------------------------------------------------------
# Cleaner selectors
# ---------------------------------------------------------------------------

def choose_net_learned(instance, paths, cleaner) -> int:
    return rank_nets_learned(instance, paths, cleaner)[0]


def rank_nets_learned(instance, paths, cleaner) -> List[int]:
    """Return net indices sorted by the cleaner's predicted score, best first.
    Used by the pipeline to skip recently-rejected nets."""
    candidates = [evaluate_net_reroute(instance, paths, i) for i in range(instance.num_nets)]
    scored = []
    for cand in candidates:
        feat = build_cleaner_features(instance, paths, int(cand["net_idx"]), candidate=cand)
        score = float(cleaner.forward(feat[None])[0])
        scored.append((score, int(cand["net_idx"])))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [nid for _, nid in scored]


def choose_net_hybrid(instance, paths, cleaner) -> int:
    """CNN score used as a tie-break over the oracle rank key. Safer than pure
    learned selection because it can never pick a net whose reroute is strictly
    worse than another's."""
    candidates = [evaluate_net_reroute(instance, paths, i) for i in range(instance.num_nets)]
    model_scores = []
    for cand in candidates:
        feat = build_cleaner_features(instance, paths, int(cand["net_idx"]), candidate=cand)
        model_scores.append(float(cleaner.forward(feat[None])[0]))
    ms = np.array(model_scores, dtype=np.float32)
    if ms.std() > 1e-6:
        ms = (ms - ms.mean()) / (ms.std() + 1e-6)
    else:
        ms = ms * 0
    scored = [(float(cand["score"]) + 1e-3 * float(ms[i]), int(cand["net_idx"]))
              for i, cand in enumerate(candidates)]
    return max(scored, key=lambda t: t[0])[1]


def choose_net_heuristic(instance, paths) -> int:
    overlap = strict_overlap_map(instance, paths)
    scores = []
    for idx, path in enumerate(paths):
        count = 0
        if path:
            for r, c in path:
                if (r, c) not in instance.pin_cells and overlap[r, c] > 0:
                    count += 1
        scores.append((count, -idx, idx))
    return max(scores, key=lambda t: (t[0], t[1]))[2]


def choose_net_random(instance, rng: random.Random) -> int:
    return rng.randrange(instance.num_nets)


def choose_net_oracle(instance, paths) -> int:
    return int(best_oracle_candidate(instance, paths)["net_idx"])


# ---------------------------------------------------------------------------
# Router modes
# ---------------------------------------------------------------------------

def reroute_astar(instance, paths, net_idx: int):
    partial = remove_net(paths, net_idx)
    src, sink = instance.nets[net_idx]
    new_path = astar(
        instance.grid_size, src, sink,
        blocked=blocked_cells_for_target(instance, partial, net_idx),
    )
    if new_path is None:
        return paths, False
    out = [list(p) if p else None for p in partial]
    out[net_idx] = new_path
    return out, True


def reroute_learned(instance, paths, net_idx, router, beam_width: int = 0):
    partial = remove_net(paths, net_idx)
    if beam_width <= 1:
        result = greedy_policy_rollout(router, instance, partial, net_idx)
    else:
        result = policy_beam_search(router, instance, partial, net_idx, beam_width=beam_width)
    if not result.success or result.path is None:
        # Safety net: if the policy can't get there, fall back to A* rather
        # than leave the net broken. Learning-assisted routers almost always
        # keep a classical fallback in practice.
        return reroute_astar(instance, paths, net_idx)
    out = [list(p) if p else None for p in partial]
    out[net_idx] = result.path
    return out, True


def reroute_learned_best_of_k(
    instance, paths, net_idx, router,
    beam_width: int = 10, top_k: int = 8,
    astar_fallback: bool = True,
):
    """Pick the best of K policy beam-search candidates by global overlap.

    If the policy returns no successful candidate, optionally fall back to
    A\* so the pipeline doesn't stall. A\* is a *fallback* not a *competitor* —
    if we put A\* in the candidate pool alongside policy paths, A\* tends to
    win on simple cases and we never see what the router learned.

    Tie-break order: (overlap, wirelength).
    """
    from search_utils import policy_beam_search_topk
    partial = remove_net(paths, net_idx)
    candidates = policy_beam_search_topk(
        router, instance, partial, net_idx,
        beam_width=beam_width, top_k=top_k,
    )

    scored = []  # (overlap, wire, path)
    for c in candidates:
        if not c.success or c.path is None:
            continue
        trial = [list(p) if p else None for p in partial]
        trial[net_idx] = c.path
        ov = strict_overlap_count(instance, trial)
        w = total_wirelength(trial)
        scored.append((ov, w, c.path))

    if not scored and astar_fallback:
        # Policy couldn't produce a single valid path — fall back to A* so
        # we don't leave the net unrouted.
        astar_paths, ok = reroute_astar(instance, paths, net_idx)
        if ok:
            ov = strict_overlap_count(instance, astar_paths)
            w = total_wirelength(astar_paths)
            scored.append((ov, w, astar_paths[net_idx]))

    if not scored:
        return paths, False

    scored.sort(key=lambda t: (t[0], t[1]))
    best_ov, best_w, best_path = scored[0]
    out = [list(p) if p else None for p in partial]
    out[net_idx] = best_path
    return out, True


def multi_rip_reroute(
    instance, paths, conflicting_nets: List[int], cleaner, router,
    beam_width: int = 10, top_k: int = 6,
):
    """Rip up several nets at once and re-route them in an order chosen by the
    cleaner.

    Multi-rip is needed because some overlaps cannot be cleared by rerouting a
    single net — net A and net B mutually block the only paths to each other.
    Removing both at once breaks the deadlock.

    For each net we generate candidates from BOTH the policy beam search AND
    cost-aware A* (where existing wires carry a soft penalty). This makes
    multi-rip a hybrid of learned and classical routing — the policy proposes
    paths it learned were good, A* proposes the global shortest-with-penalty
    path, and we keep whichever produces the lowest residual overlap.
    """
    if not conflicting_nets:
        return paths, False

    # Score the conflicting nets with the cleaner so the most valuable one
    # picks its path first (least constrained).
    scored = []
    for idx in conflicting_nets:
        cand = evaluate_net_reroute(instance, paths, idx)
        feat = build_cleaner_features(instance, paths, idx, candidate=cand)
        s = float(cleaner.forward(feat[None])[0])
        scored.append((s, idx))
    scored.sort(key=lambda t: t[0], reverse=True)
    order = [idx for _, idx in scored]

    # Remove all chosen nets at once.
    cur = [list(p) if p else None for p in paths]
    for idx in order:
        cur[idx] = None

    # Build a soft cost map from the OTHER (not-being-rerouted) nets so cost-aware
    # A* prefers cells that aren't already used. This is the classical "rip-up
    # and reroute" trick combined with our learned candidates.
    gs = instance.grid_size
    cost_map = np.zeros((gs, gs), dtype=np.float32)
    for p in cur:
        if p:
            for r, c in p:
                cost_map[r, c] += 0.4

    from search_utils import policy_beam_search_topk

    for idx in order:
        partial = cur
        # Policy candidates first.
        candidates = policy_beam_search_topk(
            router, instance, partial, idx,
            beam_width=beam_width, top_k=top_k,
        )
        scored = []
        for c in candidates:
            if not c.success or c.path is None:
                continue
            trial = [list(p) if p else None for p in partial]
            trial[idx] = c.path
            ov = strict_overlap_count(instance, trial)
            w = total_wirelength(trial)
            scored.append((ov, w, c.path))

        # If the policy returned NOTHING, fall back to A* so multi-rip can
        # still complete. We do NOT mix A* into the policy's candidate pool —
        # that would erase any contribution from the learned router.
        if not scored:
            from routing_env import astar, blocked_cells_for_target
            src, sink = instance.nets[idx]
            blocked = blocked_cells_for_target(instance, partial, idx)
            astar_path = astar(gs, src, sink, blocked=blocked, cost_map=cost_map)
            if astar_path is None:
                astar_path = astar(gs, src, sink, blocked=blocked)
            if astar_path:
                trial = [list(p) if p else None for p in partial]
                trial[idx] = astar_path
                ov = strict_overlap_count(instance, trial)
                w = total_wirelength(trial)
                scored.append((ov, w, astar_path))

        if not scored:
            return paths, False
        scored.sort(key=lambda t: (t[0], t[1]))
        chosen_path = scored[0][2]

        # Update cur and the cost map for the next net.
        cur = [list(p) if p else None for p in partial]
        cur[idx] = chosen_path
        for r, c in chosen_path:
            cost_map[r, c] += 0.4

    new_ov = strict_overlap_count(instance, cur)
    new_w = total_wirelength(cur)
    cur_ov = strict_overlap_count(instance, paths)
    cur_w = total_wirelength(paths)
    if (new_ov, new_w) < (cur_ov, cur_w):
        return cur, True
    return paths, False


def find_conflicting_nets(instance, paths, max_pairs: int = 3) -> List[List[int]]:
    """For each overlap cell, find which nets pass through it. Return up to
    `max_pairs` distinct sets of mutually-conflicting nets."""
    from collections import defaultdict
    pin_owner = {}
    for i, (s, t) in enumerate(instance.nets):
        pin_owner[s] = i; pin_owner[t] = i
    cell_users = defaultdict(set)
    for idx, path in enumerate(paths):
        if not path:
            continue
        for cell in path:
            cell_users[cell].add(idx)

    conflict_sets = []
    seen = set()
    for cell, users in cell_users.items():
        # Determine the conflicting net set at this cell.
        if cell in pin_owner:
            owner = pin_owner[cell]
            foreign = users - {owner}
            if foreign:
                conf = frozenset({owner} | foreign)
            else:
                continue
        else:
            if len(users) > 1:
                conf = frozenset(users)
            else:
                continue
        if conf in seen:
            continue
        seen.add(conf)
        conflict_sets.append(sorted(conf))
        if len(conflict_sets) >= max_pairs:
            break
    return conflict_sets




def local_conflict_exact_improve(
    instance: RoutingInstance,
    paths,
    max_iters: int = 2,
    max_paths_per_net: int = 6,
    extra_margin: int = 3,
) -> Tuple[List[Optional[List[Tuple[int, int]]]], bool]:
    """Small exact local repair over conflicting 2-3 net subsets.

    This is intentionally limited to the small-grid regime used in the project
    (10x10 / 5 nets). It only runs after the learned stage has plateaued and
    searches over pair/triple conflict subsets, so it is much cheaper than a
    full exact solve but can still clear deadlocks that single-net reroute and
    greedy multi-rip miss.
    """
    cur = [list(p) if p else None for p in paths]
    cur_key = (strict_overlap_count(instance, cur), total_wirelength(cur))
    if cur_key[0] == 0:
        return cur, False

    improved_any = False
    for _ in range(max_iters):
        conflict_sets = find_conflicting_nets(instance, cur, max_pairs=12)
        involved = sorted({n for cs in conflict_sets for n in cs})
        combos: List[List[int]] = []
        combos.extend([list(cs) for cs in conflict_sets if len(cs) >= 2])
        for i in range(len(involved)):
            for j in range(i + 1, len(involved)):
                combos.append([involved[i], involved[j]])
        for i in range(len(involved)):
            for j in range(i + 1, len(involved)):
                for k in range(j + 1, len(involved)):
                    combos.append([involved[i], involved[j], involved[k]])

        seen = set()
        best_trial = None
        for subset in combos:
            key = tuple(sorted(subset))
            if len(key) < 2 or key in seen:
                continue
            seen.add(key)

            fixed = [list(p) if p else None for p in cur]
            for idx in key:
                fixed[idx] = None

            # Search all orderings of the subset with a small candidate-path cap.
            from itertools import permutations
            subset_best = None
            for order in permutations(key):
                work = [list(p) if p else None for p in fixed]

                def dfs(pos: int):
                    nonlocal subset_best, work
                    if pos == len(order):
                        ov = strict_overlap_count(instance, work)
                        w = total_wirelength(work)
                        if subset_best is None or (ov, w) < subset_best[:2]:
                            subset_best = (ov, w, [list(p) if p else None for p in work])
                        return

                    idx = int(order[pos])
                    occupied = set()
                    for j, pth in enumerate(work):
                        if j != idx and pth:
                            occupied.update(pth)

                    cand_paths = enumerate_candidate_paths(
                        instance, idx, occupied=occupied,
                        max_paths=max_paths_per_net, extra_margin=extra_margin,
                    )
                    for pth in cand_paths:
                        work[idx] = list(pth)
                        dfs(pos + 1)
                        work[idx] = None

                dfs(0)

            if subset_best is not None and subset_best[:2] < cur_key:
                if best_trial is None or subset_best[:2] < best_trial[:2]:
                    best_trial = subset_best
                    if best_trial[0] == 0:
                        break

        if best_trial is None:
            break
        cur = [list(p) if p else None for p in best_trial[2]]
        cur_key = best_trial[:2]
        improved_any = True
        if cur_key[0] == 0:
            break

    return cur, improved_any

# ---------------------------------------------------------------------------
# Non-ML baseline: PathFinder / negotiation-based routing
# ---------------------------------------------------------------------------

def pathfinder_route(
    instance: RoutingInstance,
    max_iters: int = 20,
    h_growth: float = 0.4,
    p_base: float = 1.0,
) -> Optional[List[Optional[List[Tuple[int, int]]]]]:
    """Negotiation-based routing (PathFinder-style)."""
    gs = instance.grid_size
    history = np.zeros((gs, gs), dtype=np.float32)
    best_paths = None
    best_overlap = None
    best_wire = None

    for it in range(max_iters):
        occ = np.zeros((gs, gs), dtype=np.float32)
        order = list(range(instance.num_nets))
        np.random.default_rng(it).shuffle(order)
        tmp: List[Optional[List[Tuple[int, int]]]] = [None] * instance.num_nets
        for net_idx in order:
            present = np.clip(occ - 0.5, 0.0, None)
            cost_map = p_base * present + history
            src, sink = instance.nets[net_idx]
            blocked = set()
            for j, (s, t) in enumerate(instance.nets):
                if j != net_idx:
                    blocked.add(s); blocked.add(t)
            blocked.discard(src); blocked.discard(sink)
            path = astar(gs, src, sink, blocked=blocked, cost_map=cost_map)
            tmp[net_idx] = path
            if path:
                for r, c in path:
                    occ[r, c] += 1.0
        overused = np.clip(occ - 1.0, 0.0, None)
        history += h_growth * overused

        overlap = strict_overlap_count(instance, tmp)
        wire = total_wirelength(tmp)
        if best_overlap is None or (overlap, wire) < (best_overlap, best_wire):
            best_overlap = overlap
            best_wire = wire
            best_paths = [list(p) if p else None for p in tmp]
        if overlap == 0:
            return tmp
    return best_paths


# ---------------------------------------------------------------------------
# Method runner
# ---------------------------------------------------------------------------

def run_method(
    instance: RoutingInstance,
    method: str,
    cleaner,
    router,
    rounds: int = 5,
    beam_width: int = 4,
    rng: Optional[random.Random] = None,
    ablation: Optional[Dict[str, bool]] = None,
) -> Dict:
    """Run one method on one instance.

    `ablation` controls which advanced pipeline features are enabled when the
    ML pipeline is selected. Pass None for the full system, or a dict with
    boolean flags to disable individual components:

        gate            : if False, accept every reroute (no overlap gate)
        best_of_k       : if False, take single argmax beam path
        multi_rip       : if False, skip the multi-net rip-up phase
        rejected_skip   : if False, don't track rejected nets (let cleaner
                          re-pick the same net every round)

    Default (no ablation arg): all flags True.
    """
    abl = {"gate": True, "best_of_k": True, "multi_rip": True, "rejected_skip": True}
    if ablation:
        abl.update(ablation)
    rng = rng or random.Random(0)
    initial_paths = route_all_independent(instance)
    paths = [list(p) if p else None for p in initial_paths]
    before_overlap = strict_overlap_count(instance, paths)
    before_literal = literal_overlap_count(instance, paths)
    before_wire = total_wirelength(paths)

    chosen_nets: List[int] = []

    if method == "initial":
        pass

    elif method == "pathfinder":
        pf = pathfinder_route(instance, max_iters=15)
        if pf is not None:
            paths = pf

    elif method == "exact":
        solved = solve_instance_exact(instance, max_paths_per_net=12, extra_margin=6)
        if solved is not None:
            paths = solved

    elif method in {"random", "heuristic", "oracle_cleaner",
                     "learned_cleaner_astar_router", "astar_cleaner_learned_router",
                     "learned_cleaner_learned_router", "hybrid_cleaner_learned_router"}:
        # Track best-ever paths and skip recently-rejected nets so a single
        # bad choice doesn't get retried every round.
        best_paths = [list(p) if p else None for p in paths]
        best_ov = strict_overlap_count(instance, best_paths)
        best_w = total_wirelength(best_paths)
        rejected_nets: Set[int] = set()
        for _ in range(rounds):
            cur_ov = strict_overlap_count(instance, paths)
            if cur_ov == 0:
                break

            # Pick a net not in the rejected set (reset the set if all nets rejected).
            # If rejected_skip is OFF, we always pick the cleaner's #1 even if it
            # was just rejected — simulates the naive cleaner-router-loop.
            if method == "learned_cleaner_learned_router":
                ranking = rank_nets_learned(instance, paths, cleaner)
                if abl["rejected_skip"]:
                    chosen = next((n for n in ranking if n not in rejected_nets), ranking[0])
                else:
                    chosen = ranking[0]
            elif method == "hybrid_cleaner_learned_router":
                chosen = choose_net_hybrid(instance, paths, cleaner)
                if abl["rejected_skip"] and chosen in rejected_nets:
                    rest = [i for i in range(instance.num_nets) if i not in rejected_nets]
                    if rest:
                        chosen = rest[0]
            elif method == "random":
                chosen = choose_net_random(instance, rng)
            elif method == "heuristic":
                chosen = choose_net_heuristic(instance, paths)
            elif method == "oracle_cleaner":
                chosen = choose_net_oracle(instance, paths)
            elif method == "learned_cleaner_astar_router":
                ranking = rank_nets_learned(instance, paths, cleaner)
                if abl["rejected_skip"]:
                    chosen = next((n for n in ranking if n not in rejected_nets), ranking[0])
                else:
                    chosen = ranking[0]
            elif method == "astar_cleaner_learned_router":
                chosen = choose_net_heuristic(instance, paths)

            # Reroute according to method.
            # When best_of_k is OFF, fall back to plain greedy/beam-search reroute
            # which takes the single argmax-logprob path.
            if method in {"random", "heuristic", "oracle_cleaner", "learned_cleaner_astar_router"}:
                new_paths, _ = reroute_astar(instance, paths, chosen)
            else:
                if abl["best_of_k"]:
                    new_paths, _ = reroute_learned_best_of_k(instance, paths, chosen, router,
                                                             beam_width=beam_width + 4, top_k=6)
                else:
                    new_paths, _ = reroute_learned(instance, paths, chosen, router, beam_width=beam_width)

            chosen_nets.append(int(chosen))

            new_ov = strict_overlap_count(instance, new_paths)
            new_w = total_wirelength(new_paths)
            # Accept-only-if-better gate. When OFF, accept every reroute.
            if abl["gate"]:
                accept = (new_ov, new_w) < (cur_ov, total_wirelength(paths))
            else:
                accept = True

            if accept:
                paths = new_paths
                if abl["rejected_skip"]:
                    rejected_nets.clear()
                if (new_ov, new_w) < (best_ov, best_w):
                    best_paths = [list(p) if p else None for p in new_paths]
                    best_ov, best_w = new_ov, new_w
            else:
                rejected_nets.add(int(chosen))
                if abl["rejected_skip"] and len(rejected_nets) >= instance.num_nets:
                    break  # stuck, give up

        # Multi-rip phase: if single-net rip-up plateaued with overlap > 0,
        # try removing 2+ mutually-conflicting nets simultaneously and
        # re-routing them. Iterate until no improvement found.
        if abl["multi_rip"] and method in {"learned_cleaner_learned_router", "hybrid_cleaner_learned_router"}:
            for _multi_iter in range(3):  # multi-rip rounds (mimics PathFinder iteration)
                current_best_ov = strict_overlap_count(instance, best_paths)
                if current_best_ov == 0:
                    break

                conflict_sets = find_conflicting_nets(instance, best_paths, max_pairs=10)
                involved = set()
                for c in conflict_sets:
                    involved.update(c)
                involved = sorted(involved)

                pair_sets = []
                for i in range(len(involved)):
                    for j in range(i + 1, len(involved)):
                        pair_sets.append([involved[i], involved[j]])
                triple_sets = []
                for i in range(len(involved)):
                    for j in range(i + 1, len(involved)):
                        for k in range(j + 1, len(involved)):
                            triple_sets.append([involved[i], involved[j], involved[k]])

                all_combos = conflict_sets + pair_sets + triple_sets
                seen = set()
                made_progress_this_iter = False
                for confl in all_combos:
                    if len(confl) < 2:
                        continue
                    key = tuple(sorted(confl))
                    if key in seen:
                        continue
                    seen.add(key)
                    new_paths, improved = multi_rip_reroute(
                        instance, best_paths, list(confl), cleaner, router,
                        beam_width=beam_width + 4, top_k=6,
                    )
                    if improved:
                        new_ov = strict_overlap_count(instance, new_paths)
                        new_w = total_wirelength(new_paths)
                        if (new_ov, new_w) < (best_ov, best_w):
                            best_paths = [list(p) if p else None for p in new_paths]
                            best_ov, best_w = new_ov, new_w
                            made_progress_this_iter = True
                        if best_ov == 0:
                            break
                if not made_progress_this_iter:
                    break  # multi-rip plateaued, stop iterating

        # Return best-ever, not last.
        paths = best_paths

    elif method == "full_pipeline":
        # Stage 1: learned single-net improvement loop.
        cur = paths
        for _ in range(rounds):
            if strict_overlap_count(instance, cur) == 0:
                break
            chosen = choose_net_hybrid(instance, cur, cleaner)
            cur, _ = reroute_learned_best_of_k(
                instance, cur, chosen, router,
                beam_width=max(beam_width + 4, 8), top_k=6,
            )
            chosen_nets.append(int(chosen))
        paths = cur

        # Stage 2: learned multi-rip.
        if strict_overlap_count(instance, paths) > 0:
            for _ in range(3):
                conflict_sets = find_conflicting_nets(instance, paths, max_pairs=10)
                involved = sorted({n for cs in conflict_sets for n in cs})
                combos = [list(cs) for cs in conflict_sets if len(cs) >= 2]
                for i in range(len(involved)):
                    for j in range(i + 1, len(involved)):
                        combos.append([involved[i], involved[j]])
                for i in range(len(involved)):
                    for j in range(i + 1, len(involved)):
                        for k in range(j + 1, len(involved)):
                            combos.append([involved[i], involved[j], involved[k]])
                seen = set()
                made_progress = False
                base_key = (strict_overlap_count(instance, paths), total_wirelength(paths))
                for confl in combos:
                    key = tuple(sorted(confl))
                    if len(key) < 2 or key in seen:
                        continue
                    seen.add(key)
                    trial, ok = multi_rip_reroute(
                        instance, paths, list(key), cleaner, router,
                        beam_width=max(beam_width + 6, 10), top_k=8,
                    )
                    trial_key = (strict_overlap_count(instance, trial), total_wirelength(trial))
                    if ok and trial_key < base_key:
                        paths = trial
                        base_key = trial_key
                        made_progress = True
                        if base_key[0] == 0:
                            break
                if not made_progress or strict_overlap_count(instance, paths) == 0:
                    break

        # Stage 3: exact local repair on the remaining conflict subset(s).
        if strict_overlap_count(instance, paths) > 0 and instance.num_nets <= 5 and instance.grid_size <= 10:
            local_fixed, improved = local_conflict_exact_improve(
                instance, paths, max_iters=2, max_paths_per_net=6, extra_margin=3,
            )
            if improved and (strict_overlap_count(instance, local_fixed), total_wirelength(local_fixed)) < (strict_overlap_count(instance, paths), total_wirelength(paths)):
                paths = local_fixed

        # Stage 4: classical negotiation fallback.
        if strict_overlap_count(instance, paths) > 0:
            pf = pathfinder_route(instance, max_iters=15)
            if pf is not None and (strict_overlap_count(instance, pf), total_wirelength(pf)) < (strict_overlap_count(instance, paths), total_wirelength(paths)):
                paths = pf

        # Stage 5: full exact fallback only in the tiny-grid regime.
        if strict_overlap_count(instance, paths) > 0 and instance.num_nets <= 5 and instance.grid_size <= 10:
            solved = solve_instance_exact(instance, max_paths_per_net=10, extra_margin=4)
            if solved is not None and (strict_overlap_count(instance, solved), total_wirelength(solved)) < (strict_overlap_count(instance, paths), total_wirelength(paths)):
                paths = solved

    else:
        raise ValueError(f"Unknown method: {method}")

    after_overlap = strict_overlap_count(instance, paths)
    after_literal = literal_overlap_count(instance, paths)
    after_wire = total_wirelength(paths)

    return {
        "method": method,
        "grid_size": instance.grid_size,
        "num_nets": instance.num_nets,
        "before_overlap": before_overlap,
        "after_overlap": after_overlap,
        "before_literal_overlap": before_literal,
        "after_literal_overlap": after_literal,
        "before_wirelength": before_wire,
        "after_wirelength": after_wire,
        "success": int(after_overlap == 0),
        "paths_before": initial_paths,
        "paths_after": paths,
        "chosen_nets": chosen_nets,
    }


def evaluate_pipeline(instances, method, cleaner, router, rounds: int = 5, beam_width: int = 4,
                       keep_records: bool = False, ablation: Optional[Dict[str, bool]] = None):
    rng = random.Random(7)
    records = []
    per_grid: Dict[int, Dict] = {}
    for inst in instances:
        rec = run_method(inst, method, cleaner, router, rounds=rounds, beam_width=beam_width,
                          rng=rng, ablation=ablation)
        records.append(rec)
        k = int(inst.grid_size)
        pg = per_grid.setdefault(k, {"n": 0, "succ": 0, "ov_b": 0, "ov_a": 0, "w_b": 0, "w_a": 0})
        pg["n"] += 1
        pg["succ"] += rec["success"]
        pg["ov_b"] += rec["before_overlap"]
        pg["ov_a"] += rec["after_overlap"]
        pg["w_b"] += rec["before_wirelength"]
        pg["w_a"] += rec["after_wirelength"]
    n = max(1, len(records))
    summary = {
        "method": method,
        "cases": len(records),
        "completion_rate": sum(r["success"] for r in records) / n,
        "avg_overlap_before": float(np.mean([r["before_overlap"] for r in records])),
        "avg_overlap_after": float(np.mean([r["after_overlap"] for r in records])),
        "avg_wirelength_before": float(np.mean([r["before_wirelength"] for r in records])),
        "avg_wirelength_after": float(np.mean([r["after_wirelength"] for r in records])),
        "per_grid": {
            str(k): {
                "cases": v["n"], "completion_rate": v["succ"] / max(1, v["n"]),
                "avg_overlap_after": v["ov_a"] / max(1, v["n"]),
                "avg_wirelength_after": v["w_a"] / max(1, v["n"]),
            } for k, v in sorted(per_grid.items())
        },
    }
    if keep_records:
        summary["records"] = records
    return summary
