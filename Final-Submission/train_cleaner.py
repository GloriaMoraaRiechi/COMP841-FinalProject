"""
Train the Cleaner CNN.

Problem setup
-------------
For each training instance we have K candidate nets (where K = num_nets of that
instance). Each candidate has a target_score from the oracle (a lexicographic
ranking of the reroute outcome). We want the model to rank the candidates
correctly — i.e. assign the highest score to the net whose removal gives the
best overall outcome.

We minimise a combination of:
    - LISTWISE soft cross-entropy against a tie-aware target distribution:
        if multiple candidates share the top score, the probability mass is
        spread evenly over them
    - PAIRWISE margin loss: for every (i, j) where target[i] > target[j], we
      push s_i above s_j by at least some margin

Metrics reported:
    - top1_acc       : fraction of groups where argmax prediction is in the
                       set of optimal (tie-aware) nets
    - overlap_drop   : fraction of groups where the chosen net actually
                       reduces strict overlap (overlap decreases after reroute)
    - mrr            : mean reciprocal rank of an optimal net in the model's ranking
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from nn import Adam, mse_loss
from models import CleanerScoringCNN


# ---------------------------------------------------------------------------
# Group batching by grid size (feature tensors have different HxW per instance)
# ---------------------------------------------------------------------------

def group_by_instance(items: List[Dict]) -> List[List[Dict]]:
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for it in items:
        groups[int(it["group_id"])].append(it)
    out = []
    for gid in sorted(groups):
        g = sorted(groups[gid], key=lambda t: int(t["net_idx"]))
        out.append(g)
    return out


def bucket_by_grid_size(groups: List[List[Dict]]) -> Dict[int, List[List[Dict]]]:
    out: Dict[int, List[List[Dict]]] = defaultdict(list)
    for g in groups:
        gs = int(g[0]["grid_size"])
        out[gs].append(g)
    return out


def iter_batches(groups: List[List[Dict]], batch_groups: int, rng: np.random.Generator):
    """Yield batches of groups that all have the SAME grid size and the SAME
    number of candidates. Stack into a single big (N, C, H, W) array where
    N = batch_groups * K (candidates per group)."""
    buckets: Dict[Tuple[int, int], List[List[Dict]]] = defaultdict(list)
    for g in groups:
        buckets[(int(g[0]["grid_size"]), len(g))].append(g)

    batches = []
    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        for i in range(0, len(bucket), batch_groups):
            chunk = bucket[i:i + batch_groups]
            if len(chunk) == 0:
                continue
            batches.append(chunk)
    rng.shuffle(batches)
    for b in batches:
        yield b


def stack_group_batch(batch_groups: List[List[Dict]]):
    """Stack groups into:
        x        : (B*K, C, H, W)
        y        : (B, K)   target scores
        optimal  : (B, K)   1 on optimal items (tie-aware)
        net_idx  : (B, K)
        chosen_overlaps_before: (B,)   before-overlap of the group
        per_candidate_after_overlap: (B, K) after-overlap of each candidate
    """
    B = len(batch_groups)
    K = len(batch_groups[0])
    C, H, W = batch_groups[0][0]["x"].shape
    x = np.zeros((B * K, C, H, W), dtype=np.float32)
    y = np.zeros((B, K), dtype=np.float32)
    optimal = np.zeros((B, K), dtype=np.float32)
    net_idx = np.zeros((B, K), dtype=np.int64)
    before_overlap = np.zeros((B,), dtype=np.float32)
    after_overlap = np.zeros((B, K), dtype=np.float32)

    for b, g in enumerate(batch_groups):
        scores = np.array([float(it["target_score"]) for it in g], dtype=np.float32)
        best = scores.max()
        for k, it in enumerate(g):
            x[b * K + k] = it["x"]
            y[b, k] = scores[k]
            net_idx[b, k] = int(it["net_idx"])
            after_overlap[b, k] = float(it["meta"]["after_overlap"])
            if abs(scores[k] - best) < 1e-9:
                optimal[b, k] = 1.0
        before_overlap[b] = float(g[0]["meta"]["before_overlap"])
    return x, y, optimal, net_idx, before_overlap, after_overlap


# ---------------------------------------------------------------------------
# Losses (NumPy-only, differentiable wrt `scores`)
# ---------------------------------------------------------------------------

def listwise_soft_xent(scores: np.ndarray, optimal: np.ndarray) -> Tuple[float, np.ndarray]:
    """scores, optimal : (B, K). Returns (loss, grad wrt scores)."""
    # Stable softmax
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    target = optimal / optimal.sum(axis=1, keepdims=True).clip(min=1e-9)
    log_p = np.log(p.clip(min=1e-12))
    loss = -(target * log_p).sum(axis=1).mean()
    grad = (p - target) / scores.shape[0]
    return float(loss), grad


def pairwise_margin(scores: np.ndarray, y: np.ndarray, margin: float = 0.5) -> Tuple[float, np.ndarray]:
    """Margin hinge: for each pair (i, j) where y[b,i] > y[b,j],
    push scores[b,i] >= scores[b,j] + margin."""
    B, K = scores.shape
    # Build pair mask in a vectorised way
    y_i = y[:, :, None]      # (B, K, 1)
    y_j = y[:, None, :]      # (B, 1, K)
    better = (y_i > y_j).astype(np.float32)    # (B, K, K), 1 if i should beat j
    s_i = scores[:, :, None]
    s_j = scores[:, None, :]
    diff = s_j - s_i + margin
    active = better * (diff > 0).astype(np.float32)
    loss = (active * diff).sum() / max(1.0, better.sum())
    # grad wrt scores
    g_i = -(active).sum(axis=2)    # (B, K)   how many active pairs i is the "bigger" in
    g_j = (active).sum(axis=1)     # (B, K)
    grad = (g_i + g_j) / max(1.0, better.sum())
    return float(loss), grad


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _predict_groups(model, groups: List[List[Dict]]) -> List[np.ndarray]:
    """Run the model on each group and return a list of score vectors, one per group."""
    preds: List[np.ndarray] = []
    # Bucket by grid size for speed: we can stack same-size groups into big batches.
    buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[(int(g[0]["grid_size"]), len(g))].append(i)
    out = [None] * len(groups)
    for key, idx_list in buckets.items():
        gs, K = key
        # Stack in batches
        batch = 64
        for start in range(0, len(idx_list), batch):
            chunk = idx_list[start:start + batch]
            x = np.zeros((len(chunk) * K, groups[chunk[0]][0]["x"].shape[0], gs, gs), dtype=np.float32)
            for b, gi in enumerate(chunk):
                for k, it in enumerate(groups[gi]):
                    x[b * K + k] = it["x"]
            scores = model.forward(x).reshape(len(chunk), K)
            for b, gi in enumerate(chunk):
                out[gi] = scores[b].copy()
    return out


def evaluate_cleaner(model, items: List[Dict]) -> Dict[str, float]:
    groups = group_by_instance(items)
    if not groups:
        return {"top1_acc": float("nan"), "overlap_drop": float("nan"), "mrr": float("nan"), "count": 0}

    model_was_training = model._training
    model.eval()
    preds = _predict_groups(model, groups)
    model.train(model_was_training)

    top1_correct = 0
    overlap_drop_hit = 0
    mrr_sum = 0.0
    total = 0
    for g, ps in zip(groups, preds):
        scores = np.array([float(it["target_score"]) for it in g], dtype=np.float32)
        best = scores.max()
        optimal_idx = {k for k, s in enumerate(scores) if abs(s - best) < 1e-9}

        chosen = int(np.argmax(ps))
        if chosen in optimal_idx:
            top1_correct += 1

        before_overlap = int(g[0]["meta"]["before_overlap"])
        after = int(g[chosen]["meta"]["after_overlap"])
        if after < before_overlap:
            overlap_drop_hit += 1

        # MRR: rank of the first optimal in our ordering
        order = np.argsort(-ps)
        for rank, k in enumerate(order):
            if int(k) in optimal_idx:
                mrr_sum += 1.0 / (rank + 1)
                break
        total += 1

    return {
        "top1_acc": top1_correct / total,
        "overlap_drop": overlap_drop_hit / total,
        "mrr": mrr_sum / total,
        "count": total,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train_cleaner(
    dataset,
    out_dir: str,
    epochs: int = 30,
    batch_groups: int = 32,
    lr: float = 3e-3,
    width: int = 32,
    depth: int = 3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    train_items = dataset["train"]["cleaner"]
    val_items = dataset["val"]["cleaner"]
    test_items = dataset["test"]["cleaner"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    model = CleanerScoringCNN(
        in_channels=dataset["meta"]["cleaner_channels"],
        width=width,
        depth=depth,
    )
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_groups = group_by_instance(train_items)
    rng = np.random.default_rng(seed)

    history = {
        "config": dict(epochs=epochs, batch_groups=batch_groups, lr=lr, width=width, depth=depth),
        "epochs": [],
    }

    best_val_score = -1.0
    best_state = None
    ema_val = None  # exponential moving average of the selection score
    ema_decay = 0.5

    warmup_epochs = max(1, min(3, epochs // 5))
    base_lr = lr

    for epoch in range(1, epochs + 1):
        # Smooth LR schedule: linear warmup for the first `warmup_epochs`
        # then cosine decay down to 10% of base_lr. This removes the sharp
        # train_top1 -> val_top1 oscillation we saw previously, which came
        # from a too-large initial LR combined with hard LR halvings on
        # plateau.
        if epoch <= warmup_epochs:
            cur_lr = base_lr * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            cur_lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))
        opt.set_lr(float(cur_lr))

        model.train()
        t0 = time.time()
        losses = []
        for batch in iter_batches(train_groups, batch_groups, rng):
            x, y, optimal, _, _, _ = stack_group_batch(batch)
            B, K = y.shape

            scores_flat = model.forward(x)
            scores = scores_flat.reshape(B, K)

            loss_list, grad_list = listwise_soft_xent(scores, optimal)
            loss_pair, grad_pair = pairwise_margin(scores, y, margin=0.5)
            loss = 0.6 * loss_list + 0.4 * loss_pair
            grad = (0.6 * grad_list + 0.4 * grad_pair).reshape(-1)

            opt.zero_grad()
            model.backward(grad.astype(np.float32))
            # tighter gradient clipping for smoother updates
            total_norm = 0.0
            for p in model.parameters():
                total_norm += float((p.grad ** 2).sum())
            total_norm = total_norm ** 0.5
            if total_norm > 2.0:
                scale = 2.0 / (total_norm + 1e-8)
                for p in model.parameters():
                    p.grad *= scale
            opt.step()
            losses.append(float(loss))

        train_metrics = evaluate_cleaner(model, train_items[:3000]) if len(train_items) > 3000 else evaluate_cleaner(model, train_items)
        val_metrics = evaluate_cleaner(model, val_items)

        # Combined selection metric: 0.7 * top-1 + 0.3 * MRR. MRR is smoother
        # than top-1 because it rewards "almost got it right", so the combo
        # is less noisy between epochs than top-1 alone. We pick the best
        # model by the EMA of this combo — which stops us from latching on
        # to a single lucky epoch.
        sel = 0.7 * val_metrics["top1_acc"] + 0.3 * val_metrics["mrr"]
        ema_val = sel if ema_val is None else ema_decay * ema_val + (1 - ema_decay) * sel

        rec = dict(
            epoch=epoch,
            lr=float(cur_lr),
            loss=float(np.mean(losses)) if losses else float("nan"),
            train_top1=train_metrics["top1_acc"],
            train_overlap_drop=train_metrics["overlap_drop"],
            val_top1=val_metrics["top1_acc"],
            val_overlap_drop=val_metrics["overlap_drop"],
            val_mrr=val_metrics["mrr"],
            val_selection=float(sel),
            val_selection_ema=float(ema_val),
            seconds=time.time() - t0,
        )
        history["epochs"].append(rec)
        if verbose:
            print(f"Cleaner ep {epoch:02d} | lr {cur_lr:.4f} | loss {rec['loss']:.4f} | "
                  f"train_top1 {rec['train_top1']:.3f} | val_top1 {rec['val_top1']:.3f} | "
                  f"val_mrr {rec['val_mrr']:.3f} | sel_ema {ema_val:.3f} | {rec['seconds']:.1f}s")

        if ema_val > best_val_score + 1e-4:
            best_val_score = ema_val
            best_state = model.state_dict()

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_cleaner(model, test_items)
    history["best_val_selection_ema"] = float(best_val_score)
    history["test_top1"] = test_metrics["top1_acc"]
    history["test_overlap_drop"] = test_metrics["overlap_drop"]
    history["test_mrr"] = test_metrics["mrr"]

    # Save
    np.savez(out / "cleaner_best.npz", **{k: v for k, v in model.state_dict().items()})
    (out / "cleaner_history.json").write_text(json.dumps(history, indent=2))

    if verbose:
        print(f"\nCleaner best val selection (EMA) : {best_val_score:.3f}")
        print(f"Cleaner test top1     : {test_metrics['top1_acc']:.3f}")
        print(f"Cleaner test MRR      : {test_metrics['mrr']:.3f}")
        print(f"Cleaner test drop-hit : {test_metrics['overlap_drop']:.3f}")
    return dict(model=model, history=history, test_metrics=test_metrics)
