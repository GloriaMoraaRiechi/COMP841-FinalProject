# Using Reinforcement Learning to Improve Physical Design Routing

This project implements a reinforcement learning based routing pipeline for integrated circuit (IC) physical design. The final version uses a learned **Cleaner CNN** to identify problematic nets and a learned **Router CNN** trained with supervised imitation and REINFORCE to reroute nets on congested grid-routing problems.

The implementation is fully contained in this final package and includes the core routing environment, neural network models, training scripts, benchmark pipeline, and saved result plots.

---

## Project Goal

Physical design routing connects circuit pins using wires while reducing congestion, overlap, and unnecessary wirelength. Traditional methods such as A* and PathFinder are useful baselines, but this project explores whether a learned cooperative pipeline can improve routing completion by learning which nets to remove and how to reroute them.

The final pipeline compares:

- Initial independent A* routing
- PathFinder baseline
- Cleaner-only learned pipeline
- Router-only learned pipeline
- Full learned pipeline without RL
- Full learned pipeline with RL

---

## Final Implementation Pipeline

### 1. Dataset Generation

`dataset_generation.py` creates fixed-seed routing instances for reproducible training and testing. The final benchmark run used:

- 20,000 generated routing instances
- 10 × 10 grid size
- 5 nets per instance
- Minimum Manhattan distance of 3
- Fixed benchmark test seed
- 200 benchmark test cases

### 2. Routing Environment

`routing_env.py` defines the grid-routing environment, including:

- Grid representation
- Net and pin placement
- A* routing baseline
- Pin-aware overlap calculation
- Reward and overlap reduction logic
- Oracle rerouting utilities

### 3. Neural Network Models

`models.py` defines the learned models:

- **CleanerScoringCNN**: scores nets and selects which congested net should be removed.
- **RouterPolicyValueNet**: predicts routing actions and estimates value during routing.

The CNN components are implemented using the custom NumPy neural network framework in `nn.py`.

### 4. Cleaner Training

`train_cleaner.py` trains the Cleaner CNN using supervised learning. The Cleaner learns to identify the net that is most useful to remove for reducing congestion.

### 5. Router Training

`train_router.py` trains the Router CNN in two stages:

1. **Supervised imitation learning**, where the router learns from generated routing behavior.
2. **REINFORCE fine-tuning**, where the router is further optimized using reward feedback from full routing episodes.

### 6. Search and Rerouting

`search_utils.py` provides greedy, beam-search, and top-K search utilities. These are used to evaluate possible learned routing decisions during inference.

### 7. Full Pipeline Evaluation

`pipeline.py` combines the Cleaner and Router into a full iterative rip-up-and-reroute pipeline. `main.py` orchestrates the complete workflow:

1. Generate dataset
2. Train Cleaner CNN
3. Train Router CNN with supervised learning
4. Fine-tune Router with REINFORCE
5. Run benchmark comparisons
6. Run ablation studies
7. Save summary metrics and plots

---

## Final Results

The final benchmark used **200 fixed-seed test cases** from a 20,000-instance run.

## Results

### Completion Rate (higher is better)

- A*: 0.04  
- Pathfinder: 0.33  
- Full Pipeline: 0.45  

The learned pipeline achieves the highest completion rate, outperforming both A* and Pathfinder.

---

### Overlap (lower is better)

- A*: 4.26  
- Pathfinder: 1.05  
- Full Pipeline: 1.78  

The pipeline significantly reduces routing conflicts, though Pathfinder achieves the lowest overlap.

---

### Wirelength

- A*: 36.2  
- Pathfinder: 38.4  
- Full Pipeline: 40.2  

The learned approach results in higher wirelength, indicating a tradeoff between completion and efficiency.

---

### Model Accuracy

- Cleaner: 95.0% (test)  
- Router: 95.7% (test)  

Both models demonstrate strong predictive performance.

---

## Discussion

The results highlight a key tradeoff in routing:

- Classical methods optimize local path efficiency  
- The RL pipeline prioritizes global solvability  

The learned system is able to solve more routing instances at the cost of slightly longer paths. This makes it more suitable for complex routing scenarios where feasibility is more important than optimality.

---

## Result Files

The final result outputs are stored in `result/`.

```text
result/
├── benchmark_summary.json
└── plots/
    ├── ablation.png
    ├── benchmark_completion.png
    ├── benchmark_overlap.png
    ├── benchmark_wirelength.png
    ├── cleaner_training.png
    ├── dataset_stats.png
    ├── router_reinforcement_training_completion.png
    ├── router_reinforcement_training_overlap.png
    └── router_sl_training.png
```

These figures summarize dataset statistics, training behavior, benchmark performance, and ablation results.

---

## Repository Structure

```text
COMP841-Final_Project_v3/
├── main.py                         # Final orchestrator for training, RL, benchmarking, and plots
├── pipeline.py                     # Full learned routing pipeline and ablation methods
├── routing_env.py                  # Grid environment, A*, overlap metrics, and rewards
├── dataset_generation.py           # Reproducible dataset generation
├── models.py                       # Cleaner CNN and Router policy-value network
├── nn.py                           # Custom NumPy neural network layers and optimizer
├── search_utils.py                 # Greedy, beam, and top-K search utilities
├── train_cleaner.py                # Cleaner CNN training
├── train_router.py                 # Router supervised + REINFORCE training
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── result/
    ├── benchmark_summary.json      # Final benchmark metrics
    └── plots/                      # Generated result figures
```

---

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

The required dependencies are listed in `requirements.txt` and include:

- NumPy
- Matplotlib
- tqdm

---

## Running the Final Pipeline

To reproduce the final implementation workflow, run:

```bash
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

For a smaller test run:

```bash
python main.py --num_instances 5000 --cleaner_epochs 15 --router_epochs 15 --rl_episodes 500
```

---

## Interpretation

The results show that the learned pipeline improves routing completion compared with both initial A* routing and PathFinder on the final benchmark. PathFinder achieves the lowest residual overlap, but the full learned pipeline achieves the highest completion rate. The ablation confirms that the Cleaner CNN is the largest contributor, while the Router CNN and REINFORCE fine-tuning provide additional improvement when combined into the final pipeline.

---

## Conclusion

The Cleaner–Router reinforcement learning pipeline improves routing completion compared to traditional methods. While it introduces a wirelength tradeoff, it demonstrates strong potential for handling complex routing problems.

---

## Reference

Gandhi, U., Bustany, I., Swartz, W., & Behjat, L. (2019). *A reinforcement learning-based framework for solving physical design routing problem in the absence of large test sets*. ACM/IEEE Workshop on Machine Learning for CAD (MLCAD), 1–6.

---

## Authors

- Gloria Riechi, Computational Data Science and Engineering, North Carolina A&T State University
- Deriech Cummings, Electrical and Computer Engineering, North Carolina A&T State University
- Natnael Workneh, Electrical and Computer Engineering, North Carolina A&T State University
