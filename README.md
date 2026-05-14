# PhyGAIL

PhyGAIL is a graph-based imitation and reinforcement learning framework for post-damage UAV swarm topology recovery. This repository contains the cleaned core implementation used for training and evaluating the PhyGAIL policy, together with the retained baseline controllers used in comparison experiments.

## What is included

- PhyGAIL training pipeline
- Pretrained PhyGAIL checkpoint
- Built-in 20-agent expert datasets for train and debug
- Unified swarm simulation interface
- Retained baseline implementations:
  - `centering`
  - `HERO`
  - `SIDR`
  - `CR-MGC`
  - `DEMD`
  - `GDR-TS`
  - `MADDPG-APF`
- Basic test scripts for PhyGAIL and previous methods

## What is not included

- Large experiment logs and plotting outputs
- Full paper figures and runtime benchmark records
- Historical large database archives

The repository already includes the 20-agent expert files used by the default train and debug paths:

- `Database/expert_solutions_20_train.npz`
- `Database/expert_trajectories_20_train.npz`
- `Database/expert_solutions_20_debug.npz`
- `Database/expert_trajectories_20_debug.npz`

If you want to retrain other scales, prepare the expert files under:

- `Database/expert_solutions_<N>_train.npz`
- `Database/expert_trajectories_<N>_train.npz`

## Installation

Tested runtime environment:

- Python: `3.9.21`
- PyTorch: `2.6.0+cu124`
- CUDA runtime: `12.4`
- `torch-geometric`: `2.6.1`
- `torch-scatter`: `2.1.2+pt26cu124`
- `numpy`: `1.24.3`
- `pandas`: `2.2.3`
- `scipy`: `1.13.1`
- `matplotlib`: `3.9.2`
- `tqdm`: `4.67.1`
- `openpyxl`: `3.1.5`

Recommended setup:

```bash
conda activate <your-env>
cd PhyGAIL
pip install -r requirements.txt
```

Minimal package list:

```bash
pip install -r requirements.txt
```

## Quick Start

Test the pretrained PhyGAIL policy:

```bash
python Test_PhyGAIL.py --num_agents 20
```

Test one retained baseline:

```bash
python Test_Previous_Alg.py --algorithm centering --num_agents 20
```

Train PhyGAIL from prepared expert data:

```bash
python Train_PhyGAIL.py --training --num_agents 20 --log release
```

Run the built-in debug training path:

```bash
python Train_PhyGAIL.py --training --debug
```

Debug mode uses the bundled debug dataset, 2 epochs, 2 environments, reduced batch sizes, and writes outputs to `artifacts/debug/`.

## Verified Commands

The following commands were re-checked in the current export:

```bash
python Test_PhyGAIL.py --num_agents 20
```

Expected behavior:

- loads `models/model_best.pth`
- runs one recovery episode
- prints connectivity, recovery steps, subnet count, and average collisions

```bash
python Train_PhyGAIL.py --training --debug
```

Expected behavior:

- uses the bundled debug dataset in `Database/`
- runs `2` epochs with `2` parallel environments
- writes outputs to `artifacts/debug/`

Expected debug artifacts:

- `artifacts/debug/debug.log`
- `artifacts/debug/best_model.pth`
- `artifacts/debug/training_history.npz`
- `artifacts/debug/training_curves.png`
- `artifacts/debug/visualizations/*.png`

## Repository Structure

```text
PhyGAIL/
├── Configurations.py            # Global experiment and simulator configuration
├── Environment.py               # Damage and swarm environment dynamics
├── Utils.py                     # Connectivity and graph utility functions
├── Swarm.py                     # Unified controller interface for PhyGAIL and baselines
├── Train_PhyGAIL.py             # Main training entry for PhyGAIL
├── Test_PhyGAIL.py              # Quick test entry for the pretrained PhyGAIL model
├── Test_Previous_Alg.py         # Quick test entry for retained previous algorithms
├── PhyGAIL_Algorithm/           # Core PhyGAIL model, environment, trainer, and dataloader
├── Previous_Algorithm/          # Retained baseline implementations
├── Configurations/              # Initial swarm topology files
├── models/                      # Pretrained checkpoints
├── Database/                    # Built-in 20-agent train/debug expert data
└── artifacts/                   # Training outputs and generated checkpoints
```

## Main Components

- `PhyGAIL_Algorithm/PhyGAIL_network.py`
  - Physics-informed graph encoder
  - Actor, critic, and discriminator definitions
- `PhyGAIL_Algorithm/PhyGAIL_framework.py`
  - PPO + PhyGAIL training loop
- `PhyGAIL_Algorithm/PhyGAIL_dataloader.py`
  - Subnet graph construction and expert-data loading
- `Swarm.py`
  - Unified simulation-time action interface for:
    - `PhyGAIL`
    - `centering`
    - `HERO`
    - `SIDR`
    - `CR-MGC`
    - `DEMD`
    - `GDR-TS`
    - `MADDPG-APF`

## Notes

- The default pretrained checkpoint path is `models/model_best.pth`.
- The lightweight pretrained actor path is `models/actor_best.pth`.
- Training outputs are written to `artifacts/`.
- GPU selection follows `CUDA_VISIBLE_DEVICES` if it is already set in the shell.
- `CR-MGC` is retained with `use_meta=False`, so the exported repository does not require meta-parameter files.
