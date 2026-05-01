# Using Reinforcement Learning to improve Physical Design Routing

CNN Cleaner + REINFORCE-trained Router pipeline for grid-based routing,
benchmarked against A\* and PathFinder.

---

## Setup

### Requirements
- Python 3.9 or newer
- ~1 GB free disk space (for dataset cache + checkpoints)

## NB: The dataset could be found here: 
https://drive.google.com/file/d/17oGLWi4qBNZigUBn_iJnUQwTvwczLG4M/view?usp=drive_link


### Install
```bash
# 1. Clone or unzip the project, then enter the directory
cd COMP841-FinalProject

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate         # Linux / macOS
# venv\Scripts\activate          # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

The only dependencies are `numpy`, `matplotlib`, `networkx`, and `scipy`.
There is **no PyTorch / TensorFlow / GPU requirement** — the project
implements its CNNs from scratch using NumPy.

---

## Test (uses pre-trained checkpoints in `results/checkpoints/`)

The package ships with trained checkpoints from a 20,000-instance training
run. To verify everything works without retraining:

### 1. Solve a single 10x10 / 5-net problem and render the figure

```bash
python solve_and_visualize.py \
    --grid_size 10 \
    --nets_json '[[[1,1],[8,8]],[[1,8],[8,1]],[[2,3],[7,3]],[[3,6],[7,6]],[[4,2],[4,7]]]' \
    --cleaner_ckpt results/checkpoints/cleaner_best.npz \
    --router_ckpt  results/checkpoints/router_production.npz \
    --method learned_cleaner_learned_router \
    --rounds 5 \
    --beam_width 4 \
    --out results/case_1.png \
    --save_json
```

After the command finishes (~10 seconds), look at:

- **`results/case_1.png`** — side-by-side "Before vs After" rendering. The
  left panel shows the initial A\* routing (with overlapping cells shaded
  pink). The right panel shows the cleaned routing after the pipeline runs.
- **`results/case_1.json`** — same record in JSON: net pin coordinates, the
  paths the system chose, before/after overlap counts and wirelengths, and
  the order in which nets were ripped up.

The console will print something like:

```
Before overlap : 4    wire : 32
After  overlap : 0    wire : 38
Zero-overlap success: True
Chosen nets: [3, 1]
Figure saved to results/case_1.png
```

### 2. Compare the pipeline against A\* and PathFinder on same problem

The same script accepts other methods. Run any case three times with
different `--method` values and `--out` paths to produce side-by-side
comparison figures:

```bash
# Pure A* (independent shortest paths, no conflict handling)
python solve_and_visualize.py --grid_size 10 \
    --nets_json '[[[1,1],[8,8]],[[1,8],[8,1]],[[2,3],[7,3]],[[3,6],[7,6]],[[4,2],[4,7]]]' \
    --cleaner_ckpt results/checkpoints/cleaner_best.npz \
    --router_ckpt  results/checkpoints/router_production.npz \
    --method initial --out results/case_1_astar.png

# PathFinder negotiation-based router (classical baseline)
python solve_and_visualize.py --grid_size 10 \
    --nets_json '[[[1,1],[8,8]],[[1,8],[8,1]],[[2,3],[7,3]],[[3,6],[7,6]],[[4,2],[4,7]]]' \
    --cleaner_ckpt results/checkpoints/cleaner_best.npz \
    --router_ckpt  results/checkpoints/router_production.npz \
    --method pathfinder --out results/case_1_pathfinder.png

# Our full ML pipeline
python solve_and_visualize.py --grid_size 10 \
    --nets_json '[[[1,1],[8,8]],[[1,8],[8,1]],[[2,3],[7,3]],[[3,6],[7,6]],[[4,2],[4,7]]]' \
    --cleaner_ckpt results/checkpoints/cleaner_best.npz \
    --router_ckpt  results/checkpoints/router_production.npz \
    --method learned_cleaner_learned_router --out results/case_1_ours.png
```

---

## Where to look at the results

Pre-computed training and benchmark artifacts from the 20k-instance run
ship inside `results/`:

| File | Contents |
|---|---|
| `results/plots/cleaner_training.png` | Cleaner CNN training curves (loss, train/val accuracy) |
| `results/plots/router_sl_training.png` | Router CNN supervised-learning curves |
| `results/plots/router_rl_training.png` | Router REINFORCE training curves (4 panels) |
| `results/plots/benchmark_completion.png` | Headline result — completion rate vs A\* and PathFinder |
| `results/plots/benchmark_overlap.png` | Avg residual overlap (lower is better) |
| `results/plots/benchmark_wirelength.png` | Avg total wirelength |
| `results/plots/ablation.png` | 4-row ablation chart |
| `results/plots/dataset_stats.png` | Train / val / test sample counts |
| `results/benchmark_summary.json` | All numbers above as JSON |
| `results/checkpoints/` | Trained model weights (`cleaner_best.npz`, `router_sl.npz`, `router_production.npz`) |

**Headline result** (200 fixed-seed test instances, 10x10 grids, 5 nets):

| Method | Completion | Avg overlap | Avg wirelength |
|---|---|---|---|
| A\* (independent) | 0.045 | 4.26 | 36.2 |
| PathFinder (baseline) | 0.325 | 1.05 | 38.4 |
| **Full Pipeline (ours)** | **0.420** | 1.84 | 40.1 |

---

## Reproducing the full training run (optional, ~5 hours on a modern CPU)

If you want to retrain from scratch instead of using the supplied
checkpoints:

```bash
python main.py
```

The defaults match the supplied checkpoints: 20,000 instances, 10x10 grids,
5 nets, 30 cleaner epochs, 25 router SL epochs, 1500 RL episodes.

For a faster sanity-check run (~30 minutes, smaller numbers but correct
qualitative behaviour):

```bash
python main.py --num_instances 2000 --cleaner_epochs 10 --router_epochs 10 --rl_episodes 200
```

All training output is written to `results/` by default. Use
`--out_root some/other/dir` to change.

---

## What's in the package

| File | Purpose |
|---|---|
| `nn.py` | NumPy CNN library (Conv2D, ReLU, Linear, Adam). Gradient-checked. |
| `routing_env.py` | A\*, pin-aware overlap rule, oracle reroute. |
| `dataset_generation.py` | Fixed-seed test set (seed=12345) for reproducible benchmarks. |
| `models.py` | CleanerScoringCNN + RouterPolicyValueNet. |
| `train_cleaner.py` | Listwise + pairwise loss, warmup+cosine LR. |
| `train_router.py` | SL imitation + REINFORCE on full multi-net episodes. |
| `search_utils.py` | Greedy / beam / top-K beam search. |
| `pipeline.py` | All methods. Accept-only-if-better gate, best-of-K reroute, iterative multi-rip. |
| `plotting.py` | Training curves, benchmark bars, ablation chart. |
| `solve_and_visualize.py` | CLI to solve a custom problem and render the figure. |
| `main.py` | Orchestrator: dataset → cleaner → router SL → router RL → benchmark → ablation. |
| `resume_rl.py` | Skip cleaner/router retraining; run RL + benchmark + ablation only. |

---

## Troubleshooting

**`FileNotFoundError: results/checkpoints/cleaner_best.npz`** — the
checkpoints are inside the package's `results/checkpoints/` directory.
Make sure you `cd` into `routing_rl_package` first, so your working
directory contains both `solve_and_visualize.py` and the `results/` folder.

**`SyntaxWarning: invalid escape sequence`** — harmless. Comes from a
Python 3.12 stricter check on docstrings, does not affect output.

**The output figure looks empty / blank** — verify the `--out` directory
exists. The script does not create intermediate directories. For example,
`--out results/case_1.png` requires `results/` to exist (it does in the
shipped package).

**Slow on the first run** — generating the dataset for `main.py` takes a
few minutes for 20,000 instances. `solve_and_visualize.py` does not
generate any dataset and runs in seconds.