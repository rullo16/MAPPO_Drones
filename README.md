# MAPPO for Multi-Agent Drone Navigation

This project implements **Multi-Agent Proximal Policy Optimization (MAPPO)** to train a swarm of drones to navigate complex environments built with **Unity ML-Agents**.

The system uses a **Centralized Training, Decentralized Execution (CTDE)** architecture: agents learn with a centralized critic that sees the global state, but act using only their local visual and vector observations. To counter sparse rewards, agents are augmented with an **Intrinsic Curiosity Module (ICM)** whose weight is annealed to zero during training.

The project also includes an **MPC (MPPI) baseline** with a learned dynamics model, and **self-supervised contrastive pretraining** for the vision encoder.

## Key Features

* **Algorithm:** MAPPO with Generalized Advantage Estimation (GAE), tanh-squashed Gaussian policy (with log-prob correction), KL-based early stopping, and advantage normalization.
* **CTDE architecture:**
    * **Actor (decentralized):** acts from local camera + vector observations; parameters shared across agents.
    * **Critic (centralized):** per-agent value estimates from the concatenated observations of all agents.
* **Visual perception:** a compact strided-CNN encoder (`EfficientVisionEncoder`). The encoder is **frozen during RL** — the rollout buffer stores encoded features, so it must be pretrained first (see below). Encoded observations are `[frozen visual features | raw vector obs]`; the actor/critic learn their own vector processing end-to-end.
* **Exploration:** ICM (forward + inverse dynamics models) producing normalized, clipped intrinsic rewards, annealed to zero over the first 1M steps.
* **Curriculum learning:** 4-stage curriculum with ascending success gates and a max-steps fallback per stage, so no stage can stall the run.
* **Training health monitors:** the run aborts automatically if the policy entropy stays pinned at its maximum or the KL divergence indicates the policy has stopped moving — failure modes that previously wasted a full training run.
* **Baseline:** MPPI-based MPC agent with a learned kinematic dynamics model.
* **Monitoring:** Weights & Biases logging (`WANDB_PROJECT` / `WANDB_ENTITY` env vars).

## Project Structure

```text
.
├── MAPPO/
│   ├── train.py                    # Canonical training entry point (curriculum + no-curriculum)
│   ├── mappo_agent.py              # MAPPO agent (PPO update, checkpointing, config validation)
│   ├── models.py                   # Actor (state-independent log_std) and centralized Critic
│   ├── curiosity.py                # Intrinsic Curiosity Module (ICM)
│   ├── rollout_buffer.py           # On-policy storage with GAE
│   └── vision_encoders.py          # CNN encoder for visual observations
├── MPC/
│   └── MPC_Agent.py                # MPPI controller + learned dynamics baseline
├── tests/                          # Unit + smoke tests (pytest, no Unity required)
├── MAPPO_train.ipynb               # Thin driver notebook for MAPPO/train.py
├── MPC_train.ipynb                 # Training/evaluation for the MPC baseline
├── PretrainFeatureExtraction.ipynb # Contrastive pretraining of the vision encoder
├── Old_Approach/                   # Earlier SAC + distillation approach (kept for reference)
├── environment.yml                 # Pinned conda environment
└── Env/                            # Unity environment builds (not in the repo; see below)
```

## Installation

```bash
conda env create -f environment.yml
conda activate mappo_drone_rl
```

**Unity environment builds** are not stored in the repo. Place your builds under `./Env/` matching the paths in `MAPPO/train.py` (e.g. `./Env/Level1/DroneFlightv1`). They must expose, per agent: a camera observation, vector observations (kinematics + raycasts, 92 dims), and a 4-dim continuous action space.

## Usage

### 1. Pretrain the vision encoder (required for useful visual features)

The encoder is frozen during RL, so without pretraining the visual features are a fixed random projection.

Open `PretrainFeatureExtraction.ipynb` and run it. It pretrains the **same `EfficientVisionEncoder` architecture the agent uses** and verifies the checkpoint loads with `strict=True`. For best results, pretrain on frames collected from the Unity environment (`dataset_type="ImageFolder"`) rather than the CIFAR10 placeholder.

### 2. Train MAPPO

```bash
# With the 4-stage curriculum
python -m MAPPO.train --curriculum --no-graphics

# Without curriculum (directly on the target difficulty)
python -m MAPPO.train --env-path ./Env/FinalLevel/DroneFlightv1 --no-graphics

# Resume (restores steps, curriculum stage, optimizer, schedulers)
python -m MAPPO.train --curriculum --resume saved_models_mappo/mappo_final.pth
```

Run `python -m MAPPO.train --help` for all options (hyperparameters, save/log intervals, success threshold). Hyperparameters are validated strictly — unknown config keys raise instead of silently falling back to defaults.

### 3. Run the MPC baseline

```bash
jupyter notebook MPC_train.ipynb
```

### 4. Run the tests

```bash
pytest tests/ -q
```

The suite covers GAE math, log-prob consistency, encoder dimensions, config validation, checkpoint round-trips, and an end-to-end smoke test of two full collect→train cycles against a dummy environment. CI runs it on every push.

## Logging & Monitoring

Metrics are logged to Weights & Biases (`--no-wandb` to disable):

* **Policy/value loss, entropy, approx. KL, clip fraction, explained variance** — convergence and update health
* **Episode reward, success rate, episode length** — task performance
* **Entropy coef, curiosity coef, learning rate** — schedule state

The console log prints entropy alongside its theoretical maximum: entropy pinned at the max means the policy is pure noise, and the health monitor will abort the run rather than let it burn compute.

## Status & history

An earlier 2.39M-step training run (preserved in git history and W&B) reached only a 2% success rate; a post-mortem traced it to an oversized entropy bonus with an inverted adaptive scheduler, a curiosity module trained on temporally misaligned pairs, a vision-encoder checkpoint that silently failed to load, and a curriculum gate that could never trigger. All of these are fixed in the current code, which has not yet been validated with a full training run. Before investing GPU time, consider running ML-Agents' built-in POCA/PPO trainer on the same builds as a learnability control.
