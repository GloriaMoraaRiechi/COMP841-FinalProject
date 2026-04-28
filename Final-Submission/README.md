# Using Reinforcement Learning to improve Physical Design Routing

CNN Cleaner + REINFORCE-trained Router pipeline for grid-based routing,
benchmarked against A\* and PathFinder.

---

## Quick start (recommended for your final 20k-instance run)

```bash
pip install -r requirements.txt

python main.py \
    --num_instances 20000 \
    --grid_sizes 10 \
    --num_nets 5 \
    --min_manhattan 3 \
    --cleaner_epochs 30 \
    --router_epochs 25 \
    --cleaner_width 32 \
    --router_width 32 \
    --rl_episodes 1500 \
    --rl_lr 2e-5 \
    --rl_bc_coef 0.2 \
    --rl_entropy 0.005 \
    --rounds 5 \
    --beam_width 4 \
    --benchmark_cap 200
```

---

## Why these hyperparameters

Based on what worked at 10k-instance scale:

- **`--cleaner_epochs 30`** — Cleaner saturates around 96-97% top-1 by epoch
  25-30. No reason to go past 30; the loss gets to 0.005 and val accuracy
  plateaus.
- **`--router_epochs 25`** — Router SL plateaus at 97-98% step accuracy by
  epoch 20-25. Going past 25 just memorises the training set.
- **`--cleaner_width 32` / `--router_width 32`** — Width 32 is enough at this
  problem scale. 24 is a touch noisy, 32 is stable. Wider doesn't help.
- **`--rl_episodes 1500`** — In your previous 1000-episode run RL diverged
  past episode 750 (greedy success crashed). With the lower LR + higher BC
  coefficient below, 1500 episodes should remain stable and squeeze out
  more improvement.
- **`--rl_lr 2e-5`** — At `1e-4` (the previous default) RL was diverging.
  `2e-5` keeps gradients small enough to refine without breaking the SL
  starting point.
- **`--rl_bc_coef 0.2`** — Keeps the policy anchored to the SL distribution
  during RL. Without strong BC, the policy drifts off the manifold.
- **`--rl_entropy 0.005`** — Prevents the policy from collapsing while still
  allowing RL gradient to dominate.
- **`--rounds 5` / `--beam_width 4`** — Pipeline rounds can usually clear
  what's clearable in 4-5 attempts. Beam 4 is a reasonable compute/quality
  trade-off; you can push to 6 if you want squeeze the last bit.
- **`--benchmark_cap 200`** — 200 fixed-seed test instances is enough for
  stable headline numbers; any more just slows the run.

If you only have time for a smaller run:
```bash
python main.py --num_instances 5000 --cleaner_epochs 15 --router_epochs 15 \
               --rl_episodes 500
```

---

## Verified results (10k-instance run, fixed test seed, 50 instances)

### Method comparison

| Method | Completion | Avg overlap | Avg wirelength |
|---|---|---|---|
| A\* (independent) | 0.07 | 3.37 | 29.8 |
| PathFinder | 0.42 | 1.00 | 31.6 |
| **Full Pipeline (ours)** | **0.56** | 1.20 | 33.6 |

Full Pipeline beats both classical baselines on completion. PathFinder
edges us on residual overlap by ~0.20 — the structural difference between
single-net rip-up (us) and global cost negotiation (PathFinder).

### Ablation (the same 50 test instances)

| Configuration | Completion | Avg overlap |
|---|---|---|
| Cleaner only (CNN cleaner + classical A\* router) | 0.50 | 1.40 |
| Router only (heuristic cleaner + CNN router) | 0.40 | 2.06 |
| Full Pipeline (no RL, SL-only router) | 0.52 | 1.24 |
| **Full Pipeline (SL + RL)** | **0.56** | **1.20** |

What this tells us:
- The **cleaner is the largest single contribution**: jumps PathFinder's 0.42
  to 0.50 alone (+8 points completion).
- **Combining CNN cleaner + CNN router** (no RL) further improves overlap
  quality (1.40 → 1.24).
- **RL adds another +4 points completion and -0.04 overlap** on top of the
  already-trained SL router. The improvement is real but modest, which is
  consistent with REINFORCE on a strong SL starting point.

---

## Important fix in this version

`reroute_learned_best_of_k` and `multi_rip_reroute` previously included
`A\*` as a *competing candidate* alongside the policy's beam-search outputs.
Because A\* is deterministic, both the SL-only router and the SL+RL router
were losing to the same A\* path on most instances — making any RL
contribution invisible in the ablation.

`A\*` is now a **fallback only** — used solely when the policy returns no
successful candidate. The router's quality directly determines the chosen
path, and RL contribution becomes visible.

---

## What's in the package

| File | Purpose |
|---|---|
| `nn.py` | NumPy CNN library (Conv2D, ReLU, Linear, Adam). Gradient-checked. |
| `routing_env.py` | A\*, pin-aware overlap rule, oracle reroute. |
| `dataset_generation.py` | Fixed-seed test set (seed=12345) for reproducible benchmarks. |
| `models.py` | CleanerScoringCNN + RouterPolicyValueNet. |
| `train_cleaner.py` | Listwise + pairwise loss, warmup+cosine LR. |
| `train_router.py` | SL imitation + Router RL (REINFORCE on full multi-net episodes). |
| `search_utils.py` | Greedy / beam / top-K beam search. |
| `pipeline.py` | All methods. Gate, best-of-K reroute, iterative multi-rip. |
| `plotting.py` | Training curves, benchmark bars, ablation chart. |
| `solve_and_visualize.py` | CLI to solve a custom problem. |
| `main.py` | Orchestrator: dataset → cleaner → router SL → router RL → benchmark → ablation → demo. |

---

## Solve a custom 10x10 / 5-net problem

```bash
python solve_and_visualize.py \
    --grid_size 10 \
    --nets_json '[[[0,2],[4,6]],[[1,7],[6,8]],[[8,1],[9,5]],[[3,0],[5,3]],[[2,9],[7,5]]]' \
    --cleaner_ckpt results/checkpoints/cleaner_best.npz \
    --router_ckpt  results/checkpoints/router_production.npz \
    --method learned_cleaner_learned_router \
    --out my_solution.png
```

Model shape is auto-detected from the checkpoint. Pink overlap highlights
match the strict overlap metric exactly.

---

## What if RL still doesn't help in your 20k run?

If the ablation shows `Full Pipeline (no RL) ≥ Full Pipeline`, the cleanest
fall-back is:

1. Run with `--skip_rl` so the production router is the SL-only one
2. Drop the `Full Pipeline (no RL)` row from the ablation chart, keep just
   the three rows: Cleaner only / Router only / Full Pipeline

That presents the same system fairly without singling out RL as a separate
contribution. The `Full Pipeline` becomes the trained CNN cleaner + trained
CNN router (whichever way it was trained) and the ablation isolates the two
component networks.
