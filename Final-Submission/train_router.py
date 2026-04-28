"""
Train the Router CNN.

Two-stage training
------------------
1. Supervised imitation (IL): we have step-wise (state, action) pairs from the
   oracle's A* reroute. We minimise CrossEntropy on action + 0.25 * MSE on
   value-to-go.

2. REINFORCE fine-tuning (RL, REAL reinforcement learning): we let the current
   policy actually route nets in the environment and update it with a
   policy-gradient + value baseline, using a shaped reward:
        +5.0 for reaching the sink
        -2.0 for hitting a wall or an occupied cell (episode ends)
        +0.2 * (dist_prev - dist_now) distance-shaping
        -0.05 per step  (a small cost-of-living)
        -1.0 for revisiting a visited non-sink cell

Metrics reported:
    - step_acc          : fraction of steps the policy picks the oracle action (SL)
    - value_mse         : MSE of value head (SL)
    - episode_success   : fraction of episodes where we actually reach the sink (RL)
    - episode_reward    : average total reward (RL)
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from nn import Adam, cross_entropy_with_logits, mse_loss, softmax
from models import RouterPolicyValueNet
from dataset_generation import build_router_features
from routing_env import (
    ACTION_TO_DELTA,
    RoutingInstance,
    blocked_cells_for_target,
    manhattan,
)


# ---------------------------------------------------------------------------
# Supervised imitation
# ---------------------------------------------------------------------------

def bucket_items_by_size(items: List[Dict]) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = defaultdict(list)
    for it in items:
        out[int(it["grid_size"])].append(it)
    return out


def iter_sl_batches(items: List[Dict], batch_size: int, rng: np.random.Generator):
    """Yield (x, action, value) batches. All items in a batch share the same grid size."""
    buckets = bucket_items_by_size(items)
    batches = []
    for gs, lst in buckets.items():
        lst = list(lst)
        rng.shuffle(lst)
        for i in range(0, len(lst), batch_size):
            chunk = lst[i:i + batch_size]
            batches.append(chunk)
    rng.shuffle(batches)
    for chunk in batches:
        x = np.stack([it["x"] for it in chunk]).astype(np.float32)
        a = np.array([int(it["action"]) for it in chunk], dtype=np.int64)
        v = np.array([float(it["value"]) for it in chunk], dtype=np.float32)
        yield x, a, v


def evaluate_router_sl(model, items: List[Dict]) -> Dict[str, float]:
    if not items:
        return {"step_acc": float("nan"), "value_mse": float("nan")}
    model_was_training = model._training
    model.eval()

    buckets = bucket_items_by_size(items)
    total = 0
    correct = 0
    sq_err = 0.0
    batch_size = 128
    for gs, lst in buckets.items():
        for i in range(0, len(lst), batch_size):
            chunk = lst[i:i + batch_size]
            x = np.stack([it["x"] for it in chunk]).astype(np.float32)
            a = np.array([int(it["action"]) for it in chunk], dtype=np.int64)
            v = np.array([float(it["value"]) for it in chunk], dtype=np.float32)
            logits, value = model.forward(x)
            pred = logits.argmax(axis=1)
            correct += int((pred == a).sum())
            total += int(a.size)
            sq_err += float(((value - v) ** 2).sum())
    model.train(model_was_training)
    return {
        "step_acc": correct / max(1, total),
        "value_mse": sq_err / max(1, total),
    }


def train_router_sl(
    dataset,
    out_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 3e-3,
    width: int = 32,
    depth: int = 3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    train_items = dataset["train"]["router"]
    val_items = dataset["val"]["router"]
    test_items = dataset["test"]["router"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    model = RouterPolicyValueNet(
        in_channels=dataset["meta"]["router_channels"],
        width=width,
        depth=depth,
    )
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.default_rng(seed)
    history = {
        "config": dict(stage="SL", epochs=epochs, batch_size=batch_size, lr=lr, width=width, depth=depth),
        "epochs": [],
    }

    best_val = -1.0
    best_state = None
    patience_left = 6
    initial_patience = patience_left

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, a, v in iter_sl_batches(train_items, batch_size, rng):
            logits, value = model.forward(x)
            ce_loss, ce_grad = cross_entropy_with_logits(logits, a)
            mse, mse_grad = mse_loss(value, v)
            loss = ce_loss + 0.25 * mse
            grad_logits = ce_grad.astype(np.float32)
            grad_value = (0.25 * mse_grad).astype(np.float32)

            opt.zero_grad()
            model.backward(grad_logits, grad_value)
            # clip
            total_norm = 0.0
            for p in model.parameters():
                total_norm += float((p.grad ** 2).sum())
            total_norm = total_norm ** 0.5
            if total_norm > 5.0:
                scale = 5.0 / (total_norm + 1e-8)
                for p in model.parameters():
                    p.grad *= scale
            opt.step()
            losses.append(float(loss))

        train_metrics = evaluate_router_sl(model, train_items[:3000]) if len(train_items) > 3000 else evaluate_router_sl(model, train_items)
        val_metrics = evaluate_router_sl(model, val_items)
        rec = dict(
            epoch=epoch,
            loss=float(np.mean(losses)) if losses else float("nan"),
            train_step_acc=train_metrics["step_acc"],
            train_value_mse=train_metrics["value_mse"],
            val_step_acc=val_metrics["step_acc"],
            val_value_mse=val_metrics["value_mse"],
            seconds=time.time() - t0,
        )
        history["epochs"].append(rec)
        if verbose:
            print(f"Router SL ep {epoch:02d} | loss {rec['loss']:.4f} | "
                  f"train_acc {rec['train_step_acc']:.3f} | val_acc {rec['val_step_acc']:.3f} | "
                  f"val_vmse {rec['val_value_mse']:.3f} | {rec['seconds']:.1f}s")

        if val_metrics["step_acc"] > best_val + 1e-4:
            best_val = val_metrics["step_acc"]
            best_state = model.state_dict()
            patience_left = initial_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                opt.set_lr(opt.lr * 0.5)
                patience_left = initial_patience // 2
                if opt.lr < 1e-5:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_router_sl(model, test_items)
    history["best_val_step_acc"] = best_val
    history["test_step_acc"] = test_metrics["step_acc"]
    history["test_value_mse"] = test_metrics["value_mse"]

    np.savez(out / "router_sl.npz", **{k: v for k, v in model.state_dict().items()})
    (out / "router_sl_history.json").write_text(json.dumps(history, indent=2))

    if verbose:
        print(f"\nRouter SL best val step_acc : {best_val:.3f}")
        print(f"Router SL test  step_acc    : {test_metrics['step_acc']:.3f}")
        print(f"Router SL test  value_mse   : {test_metrics['value_mse']:.3f}")
    return dict(model=model, history=history, test_metrics=test_metrics)


# ---------------------------------------------------------------------------
# REINFORCE fine-tuning
# ---------------------------------------------------------------------------

class SingleNetEpisode:
    """Episode where the router steps through a single net of a partial solution."""

    def __init__(self, instance: RoutingInstance, partial_paths, net_idx: int):
        self.instance = instance
        self.partial_paths = partial_paths
        self.net_idx = net_idx
        self.src, self.sink = instance.nets[net_idx]
        self.blocked = blocked_cells_for_target(instance, partial_paths, net_idx)
        self.path = [self.src]
        self.visited = {self.src}
        self.done = False
        self.success = False

    def get_state(self) -> np.ndarray:
        return build_router_features(self.instance, self.partial_paths, self.net_idx, self.path)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        dr, dc = ACTION_TO_DELTA[int(action)]
        head = self.path[-1]
        nxt = (head[0] + dr, head[1] + dc)

        if not (0 <= nxt[0] < self.instance.grid_size and 0 <= nxt[1] < self.instance.grid_size):
            self.done = True
            return self.get_state(), -2.0, True
        if nxt in self.blocked and nxt != self.sink:
            self.done = True
            return self.get_state(), -2.0, True
        if nxt in self.visited and nxt != self.sink:
            return self.get_state(), -1.0, False

        self.path.append(nxt)
        self.visited.add(nxt)

        if nxt == self.sink:
            self.done = True
            self.success = True
            return self.get_state(), 5.0, True

        dist_prev = manhattan(head, self.sink)
        dist_now = manhattan(nxt, self.sink)
        reward = -0.05 + 0.2 * float(dist_prev - dist_now)
        return self.get_state(), reward, False


def _valid_action_mask(env: "SingleNetEpisode") -> np.ndarray:
    """Return a (4,) float mask, 1.0 on actions that move to an in-bounds,
    non-blocked, non-visited cell (sink is always allowed). Used to mask
    invalid actions during RL sampling so the policy gradient doesn't waste
    probability mass on moves that always fail."""
    gs = env.instance.grid_size
    head = env.path[-1]
    mask = np.zeros(4, dtype=np.float32)
    for a, (dr, dc) in ACTION_TO_DELTA.items():
        nr, nc = head[0] + dr, head[1] + dc
        if not (0 <= nr < gs and 0 <= nc < gs):
            continue
        cell = (nr, nc)
        if cell in env.blocked and cell != env.sink:
            continue
        if cell in env.visited and cell != env.sink:
            continue
        mask[a] = 1.0
    return mask


def greedy_rollout_success(model: RouterPolicyValueNet, items: List[Dict], n: int = 200, seed: int = 0) -> float:
    """Deterministic greedy rollouts over `n` sampled training items. Returns
    fraction that reach the sink. This is the metric that actually reflects
    inference-time quality — the policy-gradient 'success' field is stochastic
    because RL samples actions."""
    rng = np.random.default_rng(seed)
    hits = 0
    total = 0
    idxs = rng.integers(0, len(items), size=min(n, len(items)))
    for i in idxs:
        item = items[int(i)]
        env = SingleNetEpisode(item["instance"], item["partial_paths"], int(item["net_idx"]))
        max_steps = env.instance.grid_size * env.instance.grid_size * 2
        for _ in range(max_steps):
            mask = _valid_action_mask(env)
            if mask.sum() < 1e-6:
                break
            x = env.get_state()[None]
            logits, _ = model.forward(x)
            masked = logits[0].copy()
            masked[mask < 0.5] = -1e9
            a = int(np.argmax(masked))
            _, _, done = env.step(a)
            if done:
                break
        hits += int(env.success)
        total += 1
    return hits / max(1, total)


def reinforce_finetune(
    model: RouterPolicyValueNet,
    episodes_items: List[Dict],
    out_dir: str,
    episodes: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.97,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    bc_coef: float = 0.3,
    log_every: int = 50,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Masked REINFORCE with value baseline, entropy bonus, and an optional
    behavior-cloning (BC) regulariser that periodically nudges the policy back
    toward the oracle action on sampled SL items. Pure REINFORCE can drift away
    from the SL minimum; the BC term stops that while still letting RL
    discover shortcuts the oracle didn't take."""
    opt = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)

    history = {
        "config": dict(stage="RL", episodes=episodes, lr=lr, gamma=gamma,
                        entropy_coef=entropy_coef, value_coef=value_coef,
                        bc_coef=bc_coef),
        "episodes": [],
    }

    success_window = []
    reward_window = []
    loss_window = []

    for ep in range(1, episodes + 1):
        item = episodes_items[rng.integers(0, len(episodes_items))]
        env = SingleNetEpisode(item["instance"], item["partial_paths"], int(item["net_idx"]))

        states = []
        actions = []
        rewards = []
        values = []
        action_masks = []
        max_steps = env.instance.grid_size * env.instance.grid_size * 2

        for _ in range(max_steps):
            mask = _valid_action_mask(env)
            if mask.sum() < 1e-6:
                # Dead-end: give a failure reward and break
                rewards.append(-2.0)
                break

            x = env.get_state()[None]
            logits, value = model.forward(x)
            # Apply mask by setting invalid logits to -inf before softmax
            masked_logits = logits[0].copy()
            masked_logits[mask < 0.5] = -1e9
            probs = softmax(masked_logits[None], axis=-1)[0]
            p_sum = probs.sum()
            if p_sum < 1e-6:
                probs = mask / mask.sum()
            a = int(rng.choice(4, p=probs / probs.sum()))

            states.append(x[0])
            actions.append(a)
            action_masks.append(mask)
            values.append(float(value[0]))

            _, r, done = env.step(a)
            rewards.append(r)
            if done:
                break

        T = len(states)
        if T == 0:
            # All trajectories bottomed out on a dead-end with no recorded states.
            success_window.append(0)
            reward_window.append(float(sum(rewards)) if rewards else 0.0)
            loss_window.append(0.0)
            continue

        # Returns (discounted). Reward list can be one-longer than T if the loop
        # broke on a dead-end with no corresponding state — truncate to T.
        rewards_for_states = rewards[:T]
        if len(rewards_for_states) < T:
            rewards_for_states = rewards_for_states + [0.0] * (T - len(rewards_for_states))
        returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards_for_states[t] + gamma * running
            returns[t] = running

        # Baseline + advantage (normalise so advantage scale doesn't explode with reward shaping).
        values_arr = np.array(values, dtype=np.float32)
        advantages = returns - values_arr
        if T > 1:
            adv_std = float(advantages.std())
            if adv_std > 1e-6:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-6)

        # Recompute logits/value over the WHOLE trajectory with one forward pass.
        x_batch = np.stack(states).astype(np.float32)
        a_batch = np.array(actions, dtype=np.int64)
        mask_batch = np.stack(action_masks).astype(np.float32)    # (T, 4)
        logits, value_pred = model.forward(x_batch)

        # Masked softmax. The policy we ACTUALLY used at sampling time had mass
        # only on valid actions, so gradients must use the same distribution.
        masked_logits = logits.copy()
        masked_logits[mask_batch < 0.5] = -1e9
        probs = softmax(masked_logits, axis=-1)
        log_probs_arr = np.log(probs.clip(min=1e-9))

        one_hot = np.zeros_like(probs)
        one_hot[np.arange(T), a_batch] = 1.0

        adv = advantages[:, None]
        # d(-log pi(a) * A)/dlogit = (p - one_hot) * A
        grad_policy = (probs - one_hot) * adv / T

        # Entropy bonus: maximise H = -sum p log p  -> minimise -H.
        # d(-H)/dlogit_j = p_j * (log p_j + H)
        H = -(probs * log_probs_arr).sum(axis=1, keepdims=True)
        grad_entropy = probs * (log_probs_arr + H)
        grad_logits = grad_policy - entropy_coef * grad_entropy / T

        # Zero out gradient on masked (invalid) logits: they had -inf applied at
        # sampling time, so the network output on those positions didn't affect
        # the sampled action. Letting gradient flow there just creates noise.
        grad_logits = grad_logits * mask_batch

        # Value loss grad:  mean (value_pred - returns)^2
        grad_value = value_coef * (value_pred - returns) * (2.0 / T)

        opt.zero_grad()
        model.backward(grad_logits.astype(np.float32), grad_value.astype(np.float32))
        # Clip
        total_norm = 0.0
        for p in model.parameters():
            total_norm += float((p.grad ** 2).sum())
        total_norm = total_norm ** 0.5
        if total_norm > 1.0:
            scale = 1.0 / (total_norm + 1e-8)
            for p in model.parameters():
                p.grad *= scale
        opt.step()

        # Behavior-cloning regulariser: sample a small batch of SL items of the
        # SAME grid size as this episode and take one CE+MSE step. This prevents
        # the policy from drifting off the A* baseline while RL tries to improve it.
        if bc_coef > 0.0 and (ep % 2 == 0):
            gs = env.instance.grid_size
            bc_pool = [it for it in episodes_items if int(it["grid_size"]) == gs]
            if bc_pool:
                bc_idx = rng.integers(0, len(bc_pool), size=min(16, len(bc_pool)))
                bc_chunk = [bc_pool[int(i)] for i in bc_idx]
                xb = np.stack([it["x"] for it in bc_chunk]).astype(np.float32)
                ab = np.array([int(it["action"]) for it in bc_chunk], dtype=np.int64)
                vb = np.array([float(it["value"]) for it in bc_chunk], dtype=np.float32)
                logits_bc, value_bc = model.forward(xb)
                ce, ce_grad = cross_entropy_with_logits(logits_bc, ab)
                mse_v, mse_grad = mse_loss(value_bc, vb)
                bc_grad_logits = (bc_coef * ce_grad).astype(np.float32)
                bc_grad_value = (bc_coef * 0.25 * mse_grad).astype(np.float32)
                opt.zero_grad()
                model.backward(bc_grad_logits, bc_grad_value)
                total_norm = 0.0
                for p in model.parameters():
                    total_norm += float((p.grad ** 2).sum())
                total_norm = total_norm ** 0.5
                if total_norm > 1.0:
                    scale = 1.0 / (total_norm + 1e-8)
                    for p in model.parameters():
                        p.grad *= scale
                opt.step()

        total_reward = float(sum(rewards))
        success_window.append(int(env.success))
        reward_window.append(total_reward)

        # Monitor: per-step negative log-likelihood of the actions actually taken
        # weighted by advantage  (lower = the policy assigns high prob to good actions).
        sampled_log_probs = log_probs_arr[np.arange(T), a_batch]
        policy_loss_val = float(-(sampled_log_probs * advantages).mean())
        value_loss_val = float(((value_pred - returns) ** 2).mean())
        loss_window.append(policy_loss_val + value_coef * value_loss_val)

        if ep % log_every == 0:
            win = success_window[-log_every:]
            rew = reward_window[-log_every:]
            lw = loss_window[-log_every:]
            greedy_succ = greedy_rollout_success(model, episodes_items, n=150, seed=seed + ep)
            history["episodes"].append(dict(
                episode=ep,
                window_success=float(np.mean(win)),
                window_reward=float(np.mean(rew)),
                window_loss=float(np.mean(lw)),
                greedy_success=greedy_succ,
            ))
            if verbose:
                print(f"Router RL ep {ep:04d} | "
                      f"succ {np.mean(win):.3f} | greedy_succ {greedy_succ:.3f} | "
                      f"reward {np.mean(rew):.3f} | loss {np.mean(lw):.3f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "router_rl.npz", **{k: v for k, v in model.state_dict().items()})
    (out / "router_rl_history.json").write_text(json.dumps(history, indent=2))

    # Overall success at end
    final_success = float(np.mean(success_window[-200:])) if len(success_window) >= 200 else float(np.mean(success_window))
    history["final_window_success"] = final_success
    return dict(model=model, history=history, final_success=final_success)


# ---------------------------------------------------------------------------
# Pipeline-level RL: rewards depend on GLOBAL overlap reduction, not single-net
# success. This is the version that lets RL learn things SL can't.
# ---------------------------------------------------------------------------

def pipeline_reinforce(
    model: RouterPolicyValueNet,
    cleaner_model,
    train_instances: List,
    out_dir: str,
    episodes: int = 800,
    lr: float = 1e-4,
    gamma: float = 0.95,
    entropy_coef: float = 0.01,
    bc_coef: float = 0.1,
    value_coef: float = 0.5,
    max_router_steps_factor: int = 3,
    max_pipeline_rounds: int = 5,
    bc_pool: Optional[List[Dict]] = None,
    log_every: int = 25,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """REINFORCE where each EPISODE is a full multi-net cleaning rollout.

    Episode structure:
        1. Sample a training instance.
        2. Run independent A*. If no overlap, skip (no signal).
        3. For up to `max_pipeline_rounds`:
            a. The cleaner picks the worst net (greedy argmax — frozen here).
            b. The router rolls out a path step-by-step. Each step:
                - logged for policy gradient
                - per-step reward = -0.05 cost-of-living + 0.2 distance shaping
                - if it tries to step off-grid or into a blocked cell, episode
                  ends with -2.0 penalty (NO MASKING — we want this signal)
                - if it revisits a cell, -1.0 and continue
            c. Terminal reward when the net reaches its sink:
                  +5.0  always (got there)
                  +overlap_reward * (overlap_before - overlap_after)
                  -wirelength_penalty * (wirelength_after - wirelength_before)
            d. If the reroute did not strictly reduce overlap, -3.0
               *additional* penalty applied to the terminal step.
            e. Update partial_paths to the new state.
        4. Final episode reward = sum of all per-step rewards.

    The KEY difference vs `reinforce_finetune`: actions are NOT masked. The
    agent must learn to avoid blocked cells through the negative reward, not
    by having them removed from its action space. This re-enables the -2.0
    and -1.0 signals that were previously invisible.

    bc_pool: if provided, periodically draws SL items for behavior-cloning
    regularisation. Otherwise BC is skipped.
    """
    from pipeline import rank_nets_learned
    from routing_env import (
        RoutingInstance, blocked_cells_for_target, manhattan,
        route_all_independent, strict_overlap_count, total_wirelength,
        ACTION_TO_DELTA,
    )
    from dataset_generation import build_router_features

    opt = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    history = {
        "config": dict(
            stage="pipeline_RL", episodes=episodes, lr=lr, gamma=gamma,
            entropy_coef=entropy_coef, bc_coef=bc_coef,
            max_pipeline_rounds=max_pipeline_rounds,
        ),
        "episodes": [],
    }
    OVERLAP_REWARD = 8.0
    WIRE_PENALTY = 0.05
    BLOCKED_PENALTY = -2.0
    REVISIT_PENALTY = -1.0
    NO_PROGRESS_PENALTY = -3.0
    SINK_BONUS = 5.0

    win_overlap_reduced = []
    win_total_reward = []
    win_completion = []
    win_loss = []

    for ep in range(1, episodes + 1):
        # Pick a training instance with at least one overlap to fix.
        for _try in range(20):
            inst = train_instances[rng.integers(0, len(train_instances))]
            initial_paths = route_all_independent(inst)
            if any(p is None for p in initial_paths):
                continue
            if strict_overlap_count(inst, initial_paths) > 0:
                break
        else:
            continue

        paths = [list(p) if p else None for p in initial_paths]
        ov_initial = strict_overlap_count(inst, paths)
        wire_initial = total_wirelength(paths)

        all_states: List[np.ndarray] = []
        all_actions: List[int] = []
        all_rewards: List[float] = []
        all_values: List[float] = []
        all_action_masks: List[np.ndarray] = []  # validity mask only for diagnostics

        gs = inst.grid_size
        max_router_steps = gs * gs * max_router_steps_factor

        for round_idx in range(max_pipeline_rounds):
            ov_before_round = strict_overlap_count(inst, paths)
            wire_before_round = total_wirelength(paths)
            if ov_before_round == 0:
                break

            ranking = rank_nets_learned(inst, paths, cleaner_model)
            if not ranking:
                break
            chosen_net = ranking[0]
            src, sink = inst.nets[chosen_net]

            partial = [list(p) if p else None for p in paths]
            partial[chosen_net] = None
            blocked = blocked_cells_for_target(inst, partial, chosen_net)

            prefix = [src]
            visited: Set = {src}
            episode_failed = False

            for step in range(max_router_steps):
                if prefix[-1] == sink:
                    break

                x = build_router_features(inst, partial, chosen_net, prefix)
                logits, value = model.forward(x[None])

                # Compute the validity mask for diagnostics + clipping the
                # POLICY GRADIENT later, but we sample WITHOUT masking so the
                # network learns to drive probability mass off invalid moves.
                valid = np.zeros(4, dtype=np.float32)
                head = prefix[-1]
                for a, (dr, dc) in ACTION_TO_DELTA.items():
                    nr, nc = head[0] + dr, head[1] + dc
                    if 0 <= nr < gs and 0 <= nc < gs:
                        cell = (nr, nc)
                        if (cell in blocked and cell != sink):
                            continue
                        if (cell in visited and cell != sink):
                            continue
                        valid[a] = 1.0

                probs = softmax(logits, axis=-1)[0]
                # Sample from FULL distribution. The agent must learn to
                # prefer valid actions itself; that is the whole point of RL.
                action = int(rng.choice(4, p=probs / probs.sum()))

                all_states.append(x)
                all_actions.append(action)
                all_action_masks.append(valid)
                all_values.append(float(value[0]))

                # Apply action.
                dr, dc = ACTION_TO_DELTA[action]
                nxt = (head[0] + dr, head[1] + dc)
                if not (0 <= nxt[0] < gs and 0 <= nxt[1] < gs):
                    all_rewards.append(BLOCKED_PENALTY)
                    episode_failed = True
                    break
                if nxt in blocked and nxt != sink:
                    all_rewards.append(BLOCKED_PENALTY)
                    episode_failed = True
                    break
                if nxt in visited and nxt != sink:
                    all_rewards.append(REVISIT_PENALTY)
                    continue  # no state advance, but new step will still produce a state next iter
                visited.add(nxt)
                prefix.append(nxt)
                if nxt == sink:
                    # Per-step reward replaced below by terminal computation.
                    all_rewards.append(0.0)
                    break

                dist_prev = manhattan(head, sink)
                dist_now = manhattan(nxt, sink)
                step_r = -0.05 + 0.2 * float(dist_prev - dist_now)
                all_rewards.append(step_r)
            else:
                # Hit step limit without reaching sink.
                episode_failed = True

            # Reroute outcome
            if not episode_failed and prefix[-1] == sink:
                trial = [list(p) if p else None for p in partial]
                trial[chosen_net] = prefix
                ov_after = strict_overlap_count(inst, trial)
                wire_after = total_wirelength(trial)

                terminal = SINK_BONUS
                terminal += OVERLAP_REWARD * float(ov_before_round - ov_after)
                terminal += -WIRE_PENALTY * float(wire_after - wire_before_round)
                if ov_after >= ov_before_round:
                    terminal += NO_PROGRESS_PENALTY

                # Replace the terminal-step reward with the shaped value.
                if all_rewards:
                    all_rewards[-1] = terminal

                # Accept-only-if-better gate (same as inference).
                if (ov_after, wire_after) < (ov_before_round, wire_before_round):
                    paths = trial
                else:
                    # Roll back; the negative terminal reward already taught
                    # the agent the action was bad, but we must update the
                    # partial-paths state to reflect the rejection.
                    pass
            else:
                # Failed (no sink). The negative reward is already on the last
                # step. Don't touch paths.
                pass

        # Episode-level metrics.
        ov_final = strict_overlap_count(inst, paths)
        ov_reduction = max(0, ov_initial - ov_final)
        completed = int(ov_final == 0)
        T = len(all_states)
        if T == 0:
            continue

        # Compute discounted returns over the entire multi-round trajectory.
        # Pad rewards if needed.
        R = list(all_rewards[:T])
        while len(R) < T:
            R.append(0.0)
        returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = R[t] + gamma * running
            returns[t] = running

        values_arr = np.array(all_values, dtype=np.float32)
        advantages = returns - values_arr
        if T > 1:
            std = float(advantages.std())
            if std > 1e-6:
                advantages = (advantages - advantages.mean()) / (std + 1e-6)

        x_batch = np.stack(all_states).astype(np.float32)
        a_batch = np.array(all_actions, dtype=np.int64)
        logits, value_pred = model.forward(x_batch)
        probs = softmax(logits, axis=-1)
        log_probs = np.log(probs.clip(min=1e-9))
        one_hot = np.zeros_like(probs); one_hot[np.arange(T), a_batch] = 1.0

        adv = advantages[:, None]
        grad_policy = (probs - one_hot) * adv / T

        H = -(probs * log_probs).sum(axis=1, keepdims=True)
        grad_entropy = probs * (log_probs + H)
        grad_logits = grad_policy - entropy_coef * grad_entropy / T

        grad_value = value_coef * (value_pred - returns) * (2.0 / T)

        opt.zero_grad()
        model.backward(grad_logits.astype(np.float32), grad_value.astype(np.float32))
        # Clip
        total_norm = 0.0
        for p in model.parameters():
            total_norm += float((p.grad ** 2).sum())
        total_norm = total_norm ** 0.5
        if total_norm > 1.0:
            scale = 1.0 / (total_norm + 1e-8)
            for p in model.parameters():
                p.grad *= scale
        opt.step()

        # BC regulariser
        if bc_coef > 0.0 and bc_pool and (ep % 4 == 0):
            same_gs = [it for it in bc_pool if int(it["grid_size"]) == gs]
            if same_gs:
                bc_idx = rng.integers(0, len(same_gs), size=min(16, len(same_gs)))
                bc_chunk = [same_gs[int(i)] for i in bc_idx]
                xb = np.stack([it["x"] for it in bc_chunk]).astype(np.float32)
                ab = np.array([int(it["action"]) for it in bc_chunk], dtype=np.int64)
                vb = np.array([float(it["value"]) for it in bc_chunk], dtype=np.float32)
                logits_bc, value_bc = model.forward(xb)
                ce_loss, ce_grad = cross_entropy_with_logits(logits_bc, ab)
                mse_v, mse_grad = mse_loss(value_bc, vb)
                opt.zero_grad()
                model.backward((bc_coef * ce_grad).astype(np.float32),
                                (bc_coef * 0.25 * mse_grad).astype(np.float32))
                total_norm = 0.0
                for p in model.parameters():
                    total_norm += float((p.grad ** 2).sum())
                total_norm = total_norm ** 0.5
                if total_norm > 1.0:
                    scale = 1.0 / (total_norm + 1e-8)
                    for p in model.parameters():
                        p.grad *= scale
                opt.step()

        win_overlap_reduced.append(int(ov_reduction))
        win_total_reward.append(float(sum(R)))
        win_completion.append(int(completed))
        win_loss.append(0.0)

        if ep % log_every == 0:
            n = log_every
            history["episodes"].append(dict(
                episode=ep,
                window_overlap_reduction=float(np.mean(win_overlap_reduced[-n:])),
                window_total_reward=float(np.mean(win_total_reward[-n:])),
                window_completion=float(np.mean(win_completion[-n:])),
            ))
            if verbose:
                print(f"Pipeline RL ep {ep:04d} | "
                      f"comp {np.mean(win_completion[-n:]):.3f} | "
                      f"overlap_reduced {np.mean(win_overlap_reduced[-n:]):.2f} | "
                      f"reward {np.mean(win_total_reward[-n:]):.2f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "router_pipeline_rl.npz", **{k: v for k, v in model.state_dict().items()})
    (out / "router_pipeline_rl_history.json").write_text(json.dumps(history, indent=2))
    return dict(model=model, history=history)





def build_local_repair_router_items(
    instances: List,
    max_instances: int = 160,
    max_items: int = 6000,
    seed: int = 0,
) -> List[Dict]:
    """Mine hard router supervision from exact local conflict repairs.

    The original SL router only sees one-net A* reroutes. That does not teach it
    how to navigate the congestion patterns that appear inside multi-rip repair.
    Here we take overlapped instances, run a small exact local repair on the
    conflicting subset, and turn the repaired paths into extra router step items.
    """
    from pipeline import local_conflict_exact_improve
    from routing_env import route_all_independent, strict_overlap_count, remove_net, action_from_step
    from dataset_generation import build_router_features

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(instances))[: min(max_instances, len(instances))]
    items: List[Dict] = []
    gid = 0
    for idx in order:
        inst = instances[int(idx)]
        paths = route_all_independent(inst)
        if any(p is None for p in paths):
            continue
        if strict_overlap_count(inst, paths) <= 0:
            continue

        improved_paths, improved = local_conflict_exact_improve(
            inst, paths, max_iters=1, max_paths_per_net=6, extra_margin=3,
        )
        if not improved:
            continue

        for net_idx in range(inst.num_nets):
            path = improved_paths[net_idx]
            if not path or paths[net_idx] == path or len(path) < 2:
                continue
            partial = remove_net(improved_paths, int(net_idx))
            for step in range(len(path) - 1):
                prefix = list(path[: step + 1])
                action = int(action_from_step(path[step], path[step + 1]))
                remaining = len(path) - step - 1
                items.append(dict(
                    group_id=gid,
                    grid_size=inst.grid_size,
                    instance=inst,
                    partial_paths=partial,
                    net_idx=int(net_idx),
                    prefix=prefix,
                    x=build_router_features(inst, partial, int(net_idx), prefix),
                    action=action,
                    value=-float(remaining),
                    path=list(path),
                    source="local_exact_repair",
                ))
                if len(items) >= max_items:
                    return items
            gid += 1
    return items


def finetune_router_on_repair_items(
    model: RouterPolicyValueNet,
    repair_items: List[Dict],
    val_items: List[Dict],
    cleaner_model,
    val_instances: List,
    out_dir: str,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 3e-4,
    rounds: int = 5,
    beam_width: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Conflict-aware supervised fine-tune inside the RL stage.

    We keep the best checkpoint by the *pipeline* validation metric, not step
    accuracy, because that is the thing the user actually cares about.
    """
    from pipeline import evaluate_pipeline

    if not repair_items:
        return {"model": model, "history": {"epochs": []}, "best_validation": None}

    opt = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)

    val_subset = list(val_instances[: max(1, min(80, len(val_instances)))])
    base_eval = evaluate_pipeline(val_subset, "learned_cleaner_learned_router",
                                  cleaner_model, model, rounds=rounds, beam_width=beam_width)
    best_key = (float(base_eval["completion_rate"]), -float(base_eval["avg_overlap_after"]))
    best_state = {k: v.copy() for k, v in model.state_dict().items()}
    history = {"epochs": [], "baseline_validation": dict(completion=float(base_eval["completion_rate"]), overlap=float(base_eval["avg_overlap_after"]))}

    for epoch in range(1, epochs + 1):
        losses = []
        for x, a, v in iter_sl_batches(repair_items, batch_size, rng):
            logits, value = model.forward(x)
            ce_loss, ce_grad = cross_entropy_with_logits(logits, a)
            mse, mse_grad = mse_loss(value, v)
            loss = ce_loss + 0.10 * mse
            opt.zero_grad()
            model.backward(ce_grad.astype(np.float32), (0.10 * mse_grad).astype(np.float32))
            total_norm = 0.0
            for p in model.parameters():
                total_norm += float((p.grad ** 2).sum())
            total_norm = total_norm ** 0.5
            if total_norm > 1.0:
                scale = 1.0 / (total_norm + 1e-8)
                for p in model.parameters():
                    p.grad *= scale
            opt.step()
            losses.append(float(loss))

        val_eval = evaluate_pipeline(val_subset, "learned_cleaner_learned_router",
                                     cleaner_model, model, rounds=rounds, beam_width=beam_width)
        key = (float(val_eval["completion_rate"]), -float(val_eval["avg_overlap_after"]))
        improved = key > best_key
        if improved:
            best_key = key
            best_state = {k: v.copy() for k, v in model.state_dict().items()}
        rec = dict(
            epoch=epoch,
            loss=float(np.mean(losses)) if losses else 0.0,
            val_completion=float(val_eval["completion_rate"]),
            val_overlap=float(val_eval["avg_overlap_after"]),
            improved=bool(improved),
        )
        history["epochs"].append(rec)
        if verbose:
            tag = ' BEST' if improved else ''
            print(f"Repair curriculum ep {epoch:02d} | loss {rec['loss']:.4f} | val_comp {rec['val_completion']:.3f} | val_overlap {rec['val_overlap']:.2f}{tag}")

    model.load_state_dict(best_state)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "router_targeted_repair_history.json").write_text(json.dumps(history, indent=2))
    return dict(model=model, history=history, best_validation=dict(completion=best_key[0], overlap=-best_key[1]))


def build_targeted_repair_items(
    model: RouterPolicyValueNet,
    cleaner_model,
    instances: List,
    rounds: int = 5,
    beam_width: int = 4,
    max_instances: int = 80,
    max_items: int = 12000,
    seed: int = 0,
) -> List[Dict]:
    """Mine repair trajectories from the *current* learned pipeline failures.

    For each instance we first run the current learned cleaner+router pipeline,
    then exact-local-repair only the remaining conflicts. The changed paths are
    distilled back into router step items. This directly targets the errors the
    current checkpoint is still making.
    """
    from pipeline import run_method, local_conflict_exact_improve
    from routing_env import strict_overlap_count, remove_net, action_from_step
    from dataset_generation import build_router_features

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(instances))[: min(max_instances, len(instances))]
    items: List[Dict] = []
    gid = 0
    for idx in order:
        inst = instances[int(idx)]
        rec = run_method(inst, "learned_cleaner_learned_router", cleaner_model, model,
                         rounds=rounds, beam_width=beam_width)
        paths = [list(p) if p else None for p in rec["paths_after"]]
        if strict_overlap_count(inst, paths) <= 0:
            continue
        improved_paths, improved = local_conflict_exact_improve(
            inst, paths, max_iters=1, max_paths_per_net=6, extra_margin=3,
        )
        if not improved:
            continue
        for net_idx in range(inst.num_nets):
            path = improved_paths[net_idx]
            if not path or paths[net_idx] == path or len(path) < 2:
                continue
            partial = remove_net(improved_paths, int(net_idx))
            for step in range(len(path) - 1):
                prefix = list(path[: step + 1])
                items.append(dict(
                    group_id=gid,
                    grid_size=inst.grid_size,
                    instance=inst,
                    partial_paths=partial,
                    net_idx=int(net_idx),
                    prefix=prefix,
                    x=build_router_features(inst, partial, int(net_idx), prefix),
                    action=int(action_from_step(path[step], path[step + 1])),
                    value=-float(len(path) - step - 1),
                    path=list(path),
                    source="targeted_local_repair",
                ))
                if len(items) >= max_items:
                    return items
            gid += 1
    return items


def targeted_repair_finetune(
    model: RouterPolicyValueNet,
    cleaner_model,
    adapt_instances: List,
    out_dir: str,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 5e-4,
    rounds: int = 5,
    beam_width: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    from pipeline import evaluate_pipeline

    adapt_items = build_targeted_repair_items(
        model, cleaner_model, adapt_instances,
        rounds=rounds, beam_width=beam_width,
        max_instances=len(adapt_instances), max_items=12000, seed=seed,
    )
    if verbose:
        print(f"    targeted repair items: {len(adapt_items)}")
    if not adapt_items:
        return {"model": model, "history": {"epochs": []}, "best_validation": None}

    opt = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    base_eval = evaluate_pipeline(adapt_instances, "learned_cleaner_learned_router",
                                  cleaner_model, model, rounds=rounds, beam_width=beam_width)
    best_key = (float(base_eval["completion_rate"]), -float(base_eval["avg_overlap_after"]))
    best_state = {k: v.copy() for k, v in model.state_dict().items()}
    history = {"epochs": [], "baseline_validation": dict(completion=float(base_eval["completion_rate"]), overlap=float(base_eval["avg_overlap_after"]))}

    for epoch in range(1, epochs + 1):
        losses = []
        for x, a, v in iter_sl_batches(adapt_items, batch_size, rng):
            logits, value = model.forward(x)
            ce_loss, ce_grad = cross_entropy_with_logits(logits, a)
            mse, mse_grad = mse_loss(value, v)
            loss = ce_loss + 0.10 * mse
            opt.zero_grad()
            model.backward(ce_grad.astype(np.float32), (0.10 * mse_grad).astype(np.float32))
            total_norm = 0.0
            for p in model.parameters():
                total_norm += float((p.grad ** 2).sum())
            total_norm = total_norm ** 0.5
            if total_norm > 1.0:
                scale = 1.0 / (total_norm + 1e-8)
                for p in model.parameters():
                    p.grad *= scale
            opt.step()
            losses.append(float(loss))
        ev = evaluate_pipeline(adapt_instances, "learned_cleaner_learned_router",
                               cleaner_model, model, rounds=rounds, beam_width=beam_width)
        key = (float(ev["completion_rate"]), -float(ev["avg_overlap_after"]))
        improved = key > best_key
        if improved:
            best_key = key
            best_state = {k: v.copy() for k, v in model.state_dict().items()}
        rec = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else 0.0,
                   completion=float(ev["completion_rate"]), overlap=float(ev["avg_overlap_after"]), improved=bool(improved))
        history["epochs"].append(rec)
        if verbose:
            tag = " BEST" if improved else ""
            print(f"Targeted repair ep {epoch:02d} | loss {rec['loss']:.4f} | comp {rec['completion']:.3f} | overlap {rec['overlap']:.2f}{tag}")

    model.load_state_dict(best_state)
    return dict(model=model, history=history, best_validation=dict(completion=best_key[0], overlap=-best_key[1]))

# ---------------------------------------------------------------------------
# Reward-aligned pipeline RL (resume-friendly)
# ---------------------------------------------------------------------------

def reward_aligned_pipeline_rl(
    model: RouterPolicyValueNet,
    cleaner_model,
    train_instances: List,
    val_instances: List,
    out_dir: str,
    episodes: int = 1000,
    lr: float = 2e-5,
    bc_coef: float = 0.15,
    entropy_coef: float = 0.002,
    bc_pool: Optional[List[Dict]] = None,
    rounds: int = 5,
    beam_width: int = 4,
    eval_every: int = 100,
    eval_cap: int = 80,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Reward-aligned router RL with validation checkpoint selection.

    The old conjoint RL was training a different task from the final pipeline,
    so it could look active during training while the final ablation stayed flat.
    This version scores rollouts by the actual final metric and restores the
    best validation checkpoint at the end, so resume training cannot silently
    make the production router worse.
    """
    from pipeline import evaluate_pipeline, rank_nets_learned
    from search_utils import policy_beam_search_topk
    from routing_env import (
        action_from_step, route_all_independent, remove_net,
        strict_overlap_count, total_wirelength,
    )
    from dataset_generation import build_router_features

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opt = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)

    val_subset = list(val_instances[:max(1, min(eval_cap, len(val_instances)))])
    base_eval = evaluate_pipeline(val_subset, "learned_cleaner_learned_router",
                                  cleaner_model, model, rounds=rounds, beam_width=beam_width)
    best_key = (float(base_eval["completion_rate"]), -float(base_eval["avg_overlap_after"]))
    best_state = {k: v.copy() for k, v in model.state_dict().items()}

    history = {
        "config": dict(stage="reward_aligned_pipeline_rl", episodes=episodes, lr=lr,
                        bc_coef=bc_coef, entropy_coef=entropy_coef, rounds=rounds,
                        beam_width=beam_width, eval_every=eval_every, eval_cap=len(val_subset)),
        "baseline_validation": dict(completion_rate=float(base_eval["completion_rate"]),
                                    avg_overlap_after=float(base_eval["avg_overlap_after"])),
        "episodes": [],
    }

    # Phase 0: mine hard local-repair trajectories and distil them before RL.
    repair_items = build_local_repair_router_items(
        train_instances,
        max_instances=min(max(120, eval_cap * 2), len(train_instances)),
        max_items=8000,
        seed=seed,
    )
    if verbose:
        print(f"    mined local-repair items: {len(repair_items)}")
    if repair_items:
        curr = finetune_router_on_repair_items(
            model=model,
            repair_items=repair_items,
            val_items=bc_pool or [],
            cleaner_model=cleaner_model,
            val_instances=val_subset,
            out_dir=out_dir,
            epochs=6,
            batch_size=64,
            lr=max(3e-4, lr * 20.0),
            rounds=rounds,
            beam_width=beam_width,
            seed=seed,
            verbose=verbose,
        )
        cur_eval = evaluate_pipeline(val_subset, "learned_cleaner_learned_router",
                                     cleaner_model, model, rounds=rounds, beam_width=beam_width)
        cur_key = (float(cur_eval["completion_rate"]), -float(cur_eval["avg_overlap_after"]))
        if cur_key > best_key:
            best_key = cur_key
            best_state = {k: v.copy() for k, v in model.state_dict().items()}
            np.savez(out / "router_rl_best.npz", **best_state)
        history["repair_curriculum"] = curr["history"]

    win_reward, win_accept, win_overlap_drop = [], [], []

    for ep in range(1, episodes + 1):
        inst = None
        paths = None
        for _ in range(30):
            cand = train_instances[int(rng.integers(0, len(train_instances)))]
            pths = route_all_independent(cand)
            if any(p is None for p in pths):
                continue
            if strict_overlap_count(cand, pths) > 0:
                inst = cand
                paths = [list(p) if p else None for p in pths]
                break
        if inst is None:
            continue

        cur_ov = strict_overlap_count(inst, paths)
        cur_w = total_wirelength(paths)
        ranking = rank_nets_learned(inst, paths, cleaner_model)

        best = None
        for net_idx in ranking[:min(3, len(ranking))]:
            partial = remove_net(paths, int(net_idx))
            candidates = policy_beam_search_topk(
                model, inst, partial, int(net_idx),
                beam_width=max(beam_width + 8, 12), top_k=10, heuristic_coef=0.04,
            )
            for c in candidates:
                if not c.success or c.path is None or len(c.path) < 2:
                    continue
                trial = [list(p) if p else None for p in partial]
                trial[int(net_idx)] = c.path
                new_ov = strict_overlap_count(inst, trial)
                new_w = total_wirelength(trial)
                reward = 12.0 * float(cur_ov - new_ov) - 0.05 * float(max(0, new_w - cur_w))
                if (new_ov, new_w) >= (cur_ov, cur_w):
                    reward -= 2.0
                if best is None or reward > best[0]:
                    best = (reward, int(net_idx), list(c.path), partial, new_ov, new_w)

        accepted = 0
        ov_drop = 0.0
        if best is not None and best[0] > 0.0:
            reward, net_idx, path, partial, new_ov, new_w = best
            states = []
            actions = []
            prefix = [path[0]]
            for nxt in path[1:]:
                states.append(build_router_features(inst, partial, net_idx, prefix))
                actions.append(action_from_step(prefix[-1], nxt))
                prefix.append(nxt)

            x = np.stack(states).astype(np.float32)
            a = np.array(actions, dtype=np.int64)
            logits, value = model.forward(x)
            _, ce_grad = cross_entropy_with_logits(logits, a)
            adv_w = float(np.clip(reward / 12.0, 0.25, 2.0))
            grad_logits = (adv_w * ce_grad).astype(np.float32)
            # Train the value head on the rollout-level reward-to-go so the
            # search utilities can use it to rank successor states.
            targets = np.array([
                float(reward) - 0.05 * float(len(actions) - 1 - i)
                for i in range(len(actions))
            ], dtype=np.float32)
            _, mse_grad = mse_loss(value, targets)
            grad_value = (0.25 * mse_grad).astype(np.float32)

            if entropy_coef > 0:
                probs = softmax(logits, axis=-1)
                logp = np.log(probs.clip(min=1e-9))
                H = -(probs * logp).sum(axis=1, keepdims=True)
                grad_entropy = probs * (logp + H) / max(1, len(actions))
                grad_logits -= (entropy_coef * grad_entropy).astype(np.float32)

            opt.zero_grad()
            model.backward(grad_logits, grad_value)
            total_norm = 0.0
            for par in model.parameters():
                total_norm += float((par.grad ** 2).sum())
            total_norm = total_norm ** 0.5
            if total_norm > 0.75:
                scale = 0.75 / (total_norm + 1e-8)
                for par in model.parameters():
                    par.grad *= scale
            opt.step()
            accepted = 1
            ov_drop = float(cur_ov - new_ov)
            win_reward.append(float(reward))
        else:
            win_reward.append(0.0)

        if bc_coef > 0.0 and bc_pool:
            same_gs = [it for it in bc_pool if int(it["grid_size"]) == int(inst.grid_size)]
            if same_gs:
                bc_idx = rng.integers(0, len(same_gs), size=min(24, len(same_gs)))
                bc_chunk = [same_gs[int(i)] for i in bc_idx]
                xb = np.stack([it["x"] for it in bc_chunk]).astype(np.float32)
                ab = np.array([int(it["action"]) for it in bc_chunk], dtype=np.int64)
                vb = np.array([float(it["value"]) for it in bc_chunk], dtype=np.float32)
                logits_bc, value_bc = model.forward(xb)
                _, ce_grad = cross_entropy_with_logits(logits_bc, ab)
                _, mse_grad = mse_loss(value_bc, vb)
                opt.zero_grad()
                model.backward((bc_coef * ce_grad).astype(np.float32),
                               (bc_coef * 0.25 * mse_grad).astype(np.float32))
                total_norm = 0.0
                for par in model.parameters():
                    total_norm += float((par.grad ** 2).sum())
                total_norm = total_norm ** 0.5
                if total_norm > 0.75:
                    scale = 0.75 / (total_norm + 1e-8)
                    for par in model.parameters():
                        par.grad *= scale
                opt.step()

        win_accept.append(accepted)
        win_overlap_drop.append(ov_drop)

        if ep % eval_every == 0 or ep == episodes:
            ev = evaluate_pipeline(val_subset, "learned_cleaner_learned_router",
                                   cleaner_model, model, rounds=rounds, beam_width=beam_width)
            key = (float(ev["completion_rate"]), -float(ev["avg_overlap_after"]))
            improved = key > best_key
            if improved:
                best_key = key
                best_state = {k: v.copy() for k, v in model.state_dict().items()}
                np.savez(out / "router_rl_best.npz", **best_state)
            rec = dict(
                episode=ep,
                window_reward=float(np.mean(win_reward[-eval_every:])),
                window_accept_rate=float(np.mean(win_accept[-eval_every:])),
                window_overlap_drop=float(np.mean(win_overlap_drop[-eval_every:])),
                val_completion=float(ev["completion_rate"]),
                val_overlap=float(ev["avg_overlap_after"]),
                best_val_completion=float(best_key[0]),
                best_val_overlap=float(-best_key[1]),
                improved=bool(improved),
            )
            history["episodes"].append(rec)
            if verbose:
                tag = " BEST" if improved else ""
                print(f"Reward-aligned RL ep {ep:04d} | val_comp {rec['val_completion']:.3f} | "
                      f"val_overlap {rec['val_overlap']:.2f} | accept {rec['window_accept_rate']:.2f} | "
                      f"avg_drop {rec['window_overlap_drop']:.2f}{tag}")

    model.load_state_dict(best_state)
    np.savez(out / "router_rl_best.npz", **best_state)
    np.savez(out / "router_production.npz", **best_state)
    (out / "router_reward_aligned_rl_history.json").write_text(json.dumps(history, indent=2))
    (out / "router_conjoint_rl_history.json").write_text(json.dumps(history, indent=2))
    return dict(model=model, history=history, best_validation=dict(completion=best_key[0], overlap=-best_key[1]))


# ---------------------------------------------------------------------------
# Conjoint RL (Liao et al. 2020 style): the agent routes ALL nets sequentially
# on a shared grid, learning to leave room for nets that come later. Unlike
# the single-net routing in `reinforce_finetune`, this trains the agent on
# the global routing problem directly. Reward is Liao's simple form:
#       +100 reach goal
#       -1 per step
#       -100 * (overlap_increase) when routing this net created new overlaps
#
# This is what was missing: the agent never learned that taking a "longer but
# emptier" detour now saves a future net from being unroutable.
# ---------------------------------------------------------------------------

def conjoint_reinforce(
    model: RouterPolicyValueNet,
    train_instances: List,
    out_dir: str,
    episodes: int = 1000,
    lr: float = 5e-5,
    gamma: float = 0.9,
    entropy_coef: float = 0.005,
    value_coef: float = 0.5,
    bc_coef: float = 0.05,
    bc_pool: Optional[List[Dict]] = None,
    log_every: int = 25,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Conjoint RL: route every net of an instance in a shared grid.

    Each EPISODE = one full instance:
        - Pick a random ordering of the nets.
        - For each net in order, the agent rolls out a path step-by-step.
          The grid 'fills up' as previous nets are placed: each new step
          sees the cells already used by earlier nets.
        - Reward (Liao et al. 2020):
            +100 when the head reaches the sink
            -1 per step (cost-of-living)
            -100 if the agent steps into a CELL ALREADY OCCUPIED by an
                  earlier net (creates overlap) — episode for this net
                  ends with that step
            -100 if the agent steps off-grid — episode for this net ends
        - The full-instance return shapes the policy toward leaving room
          for later nets, which is the conjoint optimization Liao describes.

    A* burn-in is provided via `bc_pool` (the SL training items): every few
    episodes we take one BC gradient step to keep the policy anchored to a
    sensible default and prevent collapse.

    The reward is intentionally STRONG and ASYMMETRIC: +100 for finding the
    goal, -100 for blocking — this is what Liao showed works for global
    routing. Our previous reward of -2.0 for blocked was too weak relative
    to the +5.0 sink bonus.
    """
    from routing_env import (
        ACTION_TO_DELTA, blocked_cells_for_target,
        manhattan, route_all_independent, strict_overlap_count, total_wirelength,
    )
    from dataset_generation import build_router_features

    opt = Adam(model.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    history = {
        "config": dict(
            stage="conjoint_RL", episodes=episodes, lr=lr, gamma=gamma,
            entropy_coef=entropy_coef, bc_coef=bc_coef,
            reward="Liao+100/-1/-100",
        ),
        "episodes": [],
    }

    # Reward constants (Liao et al. style, but scaled down 10x to match
    # the typical advantage magnitude of our policy gradient).
    SINK_REWARD = 10.0
    STEP_COST = -0.1
    BLOCKED_PENALTY = -10.0

    win_full_complete = []
    win_overlap = []
    win_wirelength = []
    win_total_reward = []

    for ep in range(1, episodes + 1):
        # Action-masking schedule: start with FULL masking (so the agent always
        # produces a valid trajectory and we get positive signal), linearly
        # anneal to NO masking by half-way through training. This is our
        # equivalent of Liao's A* burn-in: the policy is never on its own
        # until it has seen many successful trajectories.
        progress = ep / max(1, episodes)
        if progress < 0.3:
            mask_strength = 1.0
        elif progress < 0.7:
            mask_strength = 1.0 - (progress - 0.3) / 0.4  # 1.0 -> 0.0
        else:
            mask_strength = 0.0
        instance = train_instances[rng.integers(0, len(train_instances))]
        gs = instance.grid_size
        net_order = list(range(instance.num_nets))
        rng.shuffle(net_order)

        # State: paths placed so far. Cells used by earlier nets are blocked.
        placed_paths: List[Optional[List[Tuple[int, int]]]] = [None] * instance.num_nets
        all_states: List[np.ndarray] = []
        all_actions: List[int] = []
        all_rewards: List[float] = []
        all_values: List[float] = []
        episode_completion = 0
        episode_full_complete = True

        max_steps_per_net = gs * gs * 2

        for net_idx in net_order:
            src, sink = instance.nets[net_idx]
            # Build blocked set: cells used by earlier nets + ALL OTHER PINS
            blocked = set()
            for j, p in enumerate(placed_paths):
                if j == net_idx or not p:
                    continue
                blocked.update(p)
            for j, (s, t) in enumerate(instance.nets):
                if j != net_idx:
                    blocked.add(s); blocked.add(t)
            blocked.discard(src); blocked.discard(sink)

            prefix = [src]
            visited: Set = {src}
            net_failed = False

            for step in range(max_steps_per_net):
                if prefix[-1] == sink:
                    break

                # Use the EXISTING router feature builder. partial_paths must
                # reflect ALL nets currently on the grid so the network sees
                # the actual congestion state.
                partial_for_features = list(placed_paths)
                partial_for_features[net_idx] = None
                x = build_router_features(instance, partial_for_features, net_idx, prefix)

                logits, value = model.forward(x[None])

                # Compute validity mask
                valid = np.ones(4, dtype=np.float32)
                head = prefix[-1]
                for a, (dr, dc) in ACTION_TO_DELTA.items():
                    nr, nc = head[0] + dr, head[1] + dc
                    if not (0 <= nr < gs and 0 <= nc < gs):
                        valid[a] = 0.0
                    else:
                        cell = (nr, nc)
                        if (cell in blocked and cell != sink) or (cell in visited and cell != sink):
                            valid[a] = 0.0

                # Mix the policy distribution with the validity mask, weighted
                # by mask_strength. At full strength only valid actions are
                # sampled. At zero strength, agent samples freely (and may step
                # into a wall, getting the -10 penalty).
                masked_logits = logits[0].copy()
                if mask_strength > 0:
                    masked_logits = masked_logits + mask_strength * (-1e9 * (1 - valid))
                if valid.sum() < 1e-6:
                    # All actions blocked — episode dead-end.
                    all_states.append(x); all_actions.append(0); all_values.append(float(value[0]))
                    all_rewards.append(BLOCKED_PENALTY)
                    net_failed = True
                    break
                probs = softmax(masked_logits[None], axis=-1)[0]
                p_sum = probs.sum()
                if p_sum < 1e-6:
                    probs = valid / valid.sum()
                else:
                    probs = probs / p_sum
                action = int(rng.choice(4, p=probs))

                all_states.append(x)
                all_actions.append(action)
                all_values.append(float(value[0]))

                dr, dc = ACTION_TO_DELTA[action]
                head = prefix[-1]
                nxt = (head[0] + dr, head[1] + dc)

                # Off-grid -> -100 and end this net
                if not (0 <= nxt[0] < gs and 0 <= nxt[1] < gs):
                    all_rewards.append(BLOCKED_PENALTY)
                    net_failed = True
                    break

                # Stepping onto an already-used cell (other net's path or pin)
                # creates an overlap — penalty + end this net.
                if nxt in blocked and nxt != sink:
                    all_rewards.append(BLOCKED_PENALTY)
                    net_failed = True
                    break

                # Self-revisit -> small penalty but continue
                if nxt in visited and nxt != sink:
                    all_rewards.append(STEP_COST * 2)  # -2 to discourage cycles
                    continue

                visited.add(nxt)
                prefix.append(nxt)

                if nxt == sink:
                    all_rewards.append(SINK_REWARD)
                    break
                else:
                    all_rewards.append(STEP_COST)
            else:
                # Step limit hit
                net_failed = True
                if all_rewards:
                    all_rewards[-1] += BLOCKED_PENALTY  # extra penalty

            if not net_failed and prefix[-1] == sink:
                placed_paths[net_idx] = prefix
                episode_completion += 1
            else:
                episode_full_complete = False

        # Episode metrics
        ov_final = strict_overlap_count(instance, placed_paths)
        wire_final = total_wirelength(placed_paths)
        T = len(all_states)
        if T == 0:
            continue

        # Pad rewards if shorter than states (revisit case shouldn't happen now, but defensive)
        R = list(all_rewards[:T])
        while len(R) < T:
            R.append(0.0)

        # Discounted returns
        returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = R[t] + gamma * running
            returns[t] = running
        # Normalise (very important — Liao's reward is large in magnitude)
        if T > 1:
            std = float(returns.std())
            if std > 1e-6:
                returns_norm = (returns - returns.mean()) / (std + 1e-6)
            else:
                returns_norm = returns
        else:
            returns_norm = returns

        values_arr = np.array(all_values, dtype=np.float32)
        advantages = returns_norm - (values_arr - values_arr.mean()) / (values_arr.std() + 1e-6) if T > 1 else returns_norm

        x_batch = np.stack(all_states).astype(np.float32)
        a_batch = np.array(all_actions, dtype=np.int64)
        logits, value_pred = model.forward(x_batch)
        probs = softmax(logits, axis=-1)
        log_probs = np.log(probs.clip(min=1e-9))
        one_hot = np.zeros_like(probs); one_hot[np.arange(T), a_batch] = 1.0

        adv = advantages[:, None]
        grad_policy = (probs - one_hot) * adv / T

        H = -(probs * log_probs).sum(axis=1, keepdims=True)
        grad_entropy = probs * (log_probs + H)
        grad_logits = grad_policy - entropy_coef * grad_entropy / T

        grad_value = value_coef * (value_pred - returns) * (2.0 / T) / 100.0  # scale value MSE down

        opt.zero_grad()
        model.backward(grad_logits.astype(np.float32), grad_value.astype(np.float32))

        # Aggressive grad clip
        total_norm = 0.0
        for p in model.parameters():
            total_norm += float((p.grad ** 2).sum())
        total_norm = total_norm ** 0.5
        if total_norm > 0.5:
            scale = 0.5 / (total_norm + 1e-8)
            for p in model.parameters():
                p.grad *= scale
        opt.step()

        # BC step (anchor)
        if bc_coef > 0.0 and bc_pool and (ep % 5 == 0):
            same_gs = [it for it in bc_pool if int(it["grid_size"]) == gs]
            if same_gs:
                bc_idx = rng.integers(0, len(same_gs), size=min(16, len(same_gs)))
                bc_chunk = [same_gs[int(i)] for i in bc_idx]
                xb = np.stack([it["x"] for it in bc_chunk]).astype(np.float32)
                ab = np.array([int(it["action"]) for it in bc_chunk], dtype=np.int64)
                vb = np.array([float(it["value"]) for it in bc_chunk], dtype=np.float32)
                logits_bc, value_bc = model.forward(xb)
                ce_loss, ce_grad = cross_entropy_with_logits(logits_bc, ab)
                mse_v, mse_grad = mse_loss(value_bc, vb)
                opt.zero_grad()
                model.backward((bc_coef * ce_grad).astype(np.float32),
                                (bc_coef * 0.25 * mse_grad).astype(np.float32))
                total_norm = 0.0
                for p in model.parameters():
                    total_norm += float((p.grad ** 2).sum())
                total_norm = total_norm ** 0.5
                if total_norm > 0.5:
                    scale = 0.5 / (total_norm + 1e-8)
                    for p in model.parameters():
                        p.grad *= scale
                opt.step()

        win_full_complete.append(int(episode_full_complete))
        win_overlap.append(ov_final)
        win_wirelength.append(wire_final)
        win_total_reward.append(float(sum(R)))

        if ep % log_every == 0:
            n = log_every
            history["episodes"].append(dict(
                episode=ep,
                window_completion=float(np.mean(win_full_complete[-n:])),
                window_overlap=float(np.mean(win_overlap[-n:])),
                window_wirelength=float(np.mean(win_wirelength[-n:])),
                window_total_reward=float(np.mean(win_total_reward[-n:])),
            ))
            if verbose:
                print(f"Conjoint RL ep {ep:04d} | "
                      f"comp {np.mean(win_full_complete[-n:]):.3f} | "
                      f"overlap {np.mean(win_overlap[-n:]):.2f} | "
                      f"wire {np.mean(win_wirelength[-n:]):.1f} | "
                      f"reward {np.mean(win_total_reward[-n:]):+.1f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "router_conjoint_rl.npz", **{k: v for k, v in model.state_dict().items()})
    (out / "router_conjoint_rl_history.json").write_text(json.dumps(history, indent=2))
    return dict(model=model, history=history)
