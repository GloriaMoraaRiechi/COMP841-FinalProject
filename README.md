# RL-Based Physical Design Router

A reinforcement learning framework for integrated circuit (IC) routing based on the cooperative two-agent architecture proposed by Gandhi et al. (2019).

## Overview

Routing in physical design is the process of connecting logic gates with wires while minimizing wire length and avoiding design rule violations. This project frames routing as a two-player cooperative game between two CNN agents:

- **Cleaner** — identifies the net most responsible for routing conflicts and removes it
- **Router** — re-routes the removed net using a REINFORCE-trained policy guided by MCTS at inference

## Architecture

- 5×5 routing grid with 3 nets
- 7-channel state tensor (3 path + 3 pin + 1 congestion)
- Cleaner CNN: 4 conv blocks + dense head (1.7M params)
- Router CNN: 4 conv blocks + policy head + value head (2.5M params)
- Router trained via imitation warm-start + REINFORCE
- MCTS used at inference for lookahead planning
- Joint co-training loop alternating between agents

## Results

| Method | Completion Rate | Avg Wirelength |
|--------|----------------|----------------|
| Random Cleaner (ablation) | 66.4% | 12.18 |
| RL pre co-training | 60.9% | 11.51 |
| **RL post co-training** | **94.8%** | **11.46** |
| A* Baseline | 94.9% | 11.55 |

The RL pipeline matches A* completion rate while achieving shorter average wirelength.

## Project Structure
```
COMP841-FinalProject/
├── routing_env.py              # Grid environment, A*, state tensor, reward
├── routing_env.ipynb           # Environment setup and validation
├── dataset_generation.ipynb    # Dataset generation
├── cleaner_cnn.ipynb           # Cleaner CNN training (supervised)
├── router_reinforce.ipynb      # Router REINFORCE + MCTS
├── cotraining.ipynb            # Co-training and evaluation
├── checkpoints/
│   ├── cleaner_best.pt         # Best Cleaner CNN weights
│   ├── router_best.pt          # Best Router CNN weights (REINFORCE)
│   └── phase5_final.pt         # Final co-trained checkpoint
├── data/
│   ├── cleaner_dataset.pkl
│   ├── router_dataset.pkl
│   └── net_configs.pkl
├── requirements.txt
└── setup_comp841RL.sh
```

## Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/COMP841-FinalProject.git
cd COMP841-FinalProject

# Create environment
bash setup_comp841RL.sh

# Activate
source comp841RL/bin/activate
```

## Requirements

- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- Jupyter

See `requirements.txt` for full list.

## Reference

Gandhi, U., Bustany, I., Swartz, W., & Behjat, L. (2019). A reinforcement learning-based framework for solving physical design routing problem in the absence of large test sets. *ACM/IEEE 1st Workshop on Machine Learning for CAD (MLCAD)*, pp. 1–6.

## Authors

- Deriech Cummings II — ECE Department, NC A&T State University
- Gloria Riechi — CDSE Department, NC A&T State University  
- Natnael Workneh — ECE Department, NC A&T State University# RL-Based Physical Design Router

A reinforcement learning framework for integrated circuit (IC) routing based on the cooperative two-agent architecture proposed by Gandhi et al. (2019). Implemented as a course project for COMP841 at North Carolina Agricultural and Technical State University.

## Overview

Routing in physical design is the process of connecting logic gates with wires while minimizing wire length and avoiding design rule violations. This project frames routing as a two-player cooperative game between two CNN agents:

- **Cleaner** — identifies the net most responsible for routing conflicts and removes it
- **Router** — re-routes the removed net using a REINFORCE-trained policy guided by MCTS at inference

## Architecture

- 5×5 routing grid with 3 nets
- 7-channel state tensor (3 path + 3 pin + 1 congestion)
- Cleaner CNN: 4 conv blocks + dense head (1.7M params)
- Router CNN: 4 conv blocks + policy head + value head (2.5M params)
- Router trained via imitation warm-start + REINFORCE
- MCTS used at inference for lookahead planning
- Joint co-training loop alternating between agents

## Results

| Method | Completion Rate | Avg Wirelength |
|--------|----------------|----------------|
| Random Cleaner (ablation) | 66.4% | 12.18 |
| RL pre co-training | 60.9% | 11.51 |
| **RL post co-training** | **94.8%** | **11.46** |
| A* Baseline | 94.9% | 11.55 |

The RL pipeline matches A* completion rate while achieving shorter average wirelength.

## Project Structure
```
COMP841-FinalProject/
├── routing_env.py              # Grid environment, A*, state tensor, reward
├── routing_env.ipynb    # Environment setup and validation
├── dataset_generation.ipynb  # Dataset generation
├── cleaner_cnn.ipynb    # Cleaner CNN training (supervised)
├── router_reinforce.ipynb    # Router REINFORCE + MCTS
├── cotraining.ipynb     # Co-training and evaluation
├── checkpoints/
│   ├── cleaner_best.pt         # Best Cleaner CNN weights
│   ├── router_best.pt          # Best Router CNN weights (REINFORCE)
│   └── phase5_final.pt         # Final co-trained checkpoint
├── data/
│   ├── cleaner_dataset.pkl
│   ├── router_dataset.pkl
│   └── net_configs.pkl
├── requirements.txt
└── setup_comp841RL.sh
```

## Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/COMP841-FinalProject.git
cd COMP841-FinalProject

# Create environment
bash setup_comp841RL.sh

# Activate
source comp841RL/bin/activate
```

## Requirements

- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- Gymnasium
- Jupyter

See `requirements.txt` for full list.

## Reference

Gandhi, U., Bustany, I., Swartz, W., & Behjat, L. (2019). A reinforcement learning-based framework for solving physical design routing problem in the absence of large test sets. *ACM/IEEE 1st Workshop on Machine Learning for CAD (MLCAD)*, pp. 1–6.

## Authors

- Deriech Cummings II — ECE Department, NC A&T State University
- Gloria Riechi — CDSE Department, NC A&T State University  
- Natnael Workneh — ECE Department, NC A&T State University
