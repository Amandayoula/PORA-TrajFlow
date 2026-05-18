# TrajFlow PPO for AV2

This repository contains a compact TrajFlow + PPO pipeline for risk-aware
autonomous-vehicle trajectory planning on Argoverse 2 motion forecasting
scenarios.

The workflow is:

1. Build or load AV2 trajectory caches.
2. Use a trained TrajFlow model to precompute non-AV risk heatmaps.
3. Train a multi-scenario PPO policy with acceleration and heading-rate control.
4. Visualize one scenario or plot training reports.

The main entry point is `main.py`.

## Repository Layout 

```text
TrajFlow/
├── main.py                         # Unified CLI: precompute, train, viz, report
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Optional CUDA/Python environment
├── data/                           # Local AV2 caches and scenario files
├── datasets/
│   ├── AV2_parallel.py             # Flat per-track cache for TrajFlow
│   ├── AV2_scenarios.py            # Scenario-grouped cache for PPO
│   └── AV2_map.py                  # Lightweight AV2 map loader/fallback map
├── model/
│   ├── TrajFlow.py                 # TrajFlow wrapper and model factory
│   ├── encoder/GRU.py              # GRU encoder
│   ├── encoder/CDE.py              # CDE encoder
│   ├── flow/DNF.py                 # Discrete normalizing flow
│   ├── flow/CNF.py                 # Continuous normalizing flow
│   └── layers/                     # CNF/CDE helper layers
├── rl/
│   ├── ppo.py                      # Generic PPO return/advantage utilities
│   ├── dict_ppo.py                 # Dict-observation actor-critic and PPO update
│   └── scenario_env.py             # AV2 scenario environment, reward, PORA risk
├── scripts/
│   ├── precompute_scenario_heatmaps.py
│   ├── train_ppo_multi_scenario_accel.py
│   ├── viz_one_scenario.py
│   ├── plot_multi_scenario_ppo_report.py
│   └── filter_moving_av_scenarios.py
└── runs/
    ├── ppo_final.pt                # Final PPO policy checkpoint
    ├── ppo_final.json              # Final training/evaluation history
    └── ppo_final_report.png        # Final report plot
```

## Installation

Python 3.9+ is recommended.

```bash
cd TrajFlow
pip install -r requirements.txt
```

For CUDA environments, install a PyTorch build matching your driver first if
needed. The included `Dockerfile` provides one reproducible starting point.

## Data

The code expects AV2-style data under:

```text
data/av2_mf_tiny/
├── train/<scenario_id>/*.parquet       # Raw AV2 scenarios and map JSONs, if rebuilding
├── with_fragment_no_normalization_change_boundary_all.pt
├── scenarios_rl_moving_av_100.pt
└── scenarios_rl_moving_av_100.csv
```

Important cache types:

- `with_fragment_no_normalization_change_boundary_all.pt` is the flat per-track
  TrajFlow cache.
- `scenarios_rl_moving_av_100.pt` is the scenario-grouped PPO cache.
- `data/heatmaps_rl_s100/` is the default heatmap directory used by PPO. If it
  is missing, run the heatmap precompute step below.

Raw AV2 data is only required if you want to rebuild the caches.

## Quick Start

Visualize the final policy on one scenario:

```bash
python main.py viz \
  --policy runs/ppo_final.pt \
  --scenario_id <scenario_id> \
  --out runs/viz_one/final.png
```

For an animation:

```bash
python main.py viz \
  --policy runs/ppo_final.pt \
  --scenario_id <scenario_id> \
  --out runs/viz_one/final.gif \
  --fps 4
```

Plot the final training report:

```bash
python main.py report \
  --history runs/ppo_final.json \
  --out runs/ppo_final_report.png
```

## Goal Options

Training and visualization both support the same goal modes:

```bash
# Goal at the AV recorded endpoint, shifted laterally along the lane normal.
--goal endpoint --goal_lane_offset_m 2.0

# No goal conditioning.
--goal none

# Explicit world-frame goal.
--goal world --goal_world_x 123.0 --goal_world_y 456.0
```

Use the same goal mode at visualization time that was used during training,
because goal-conditioned and no-goal policies have different observation sizes.

## Precompute Heatmaps

PPO uses precomputed TrajFlow heatmaps for each non-AV track. The default
TrajFlow checkpoint is:

```text
trajflow_GRU_DNF_marginal_AV2_best_all.pt
```

Generate heatmaps:

```bash
python main.py precompute \
  --scenarios data/av2_mf_tiny/scenarios_rl_moving_av_100.pt \
  --out data/heatmaps_rl_s100 \
  --steps 64 \
  --device cuda
```

The model code still supports all TrajFlow combinations:

```bash
--encoder GRU|CDE
--flow DNF|CNF
```

For example:

```bash
python main.py precompute \
  --scenarios data/av2_mf_tiny/scenarios_rl_moving_av_100.pt \
  --model path/to/checkpoint.pt \
  --encoder CDE \
  --flow CNF \
  --out data/heatmaps_cde_cnf
```

## Train PPO

Train the final multi-scenario PPO policy:

```bash
python main.py train \
  --goal endpoint \
  --goal_lane_offset_m 0.0 \
  --save_policy runs/ppo_final.pt \
  --save_history runs/ppo_final.json
```

No-goal policy:

```bash
python main.py train \
  --goal none \
  --save_policy runs/ppo_no_goal.pt \
  --save_history runs/ppo_no_goal.json
```

Common hyperparameters can be passed directly through `main.py train`:

```bash
python main.py train \
  --total_updates 100 \
  --rollout_steps 1024 \
  --lambda_risk 1.0 \
  --w_goal 1.0 \
  --ttc_conflict_penalty 1.0 \
  --collision_penalty 30.0
```

`main.py train` forwards unknown options to
`scripts/train_ppo_multi_scenario_accel.py`.

## Rebuild Scenario Caches

Build a scenario-grouped cache from raw AV2 parquet files:

```bash
python -m datasets.AV2_scenarios \
  --root data/av2_mf_tiny \
  --out data/av2_mf_tiny/scenarios_rl.pt \
  --ref_cache data/av2_mf_tiny/with_fragment_no_normalization_change_boundary_all.pt
```

Filter to moving-AV scenarios:

```bash
python scripts/filter_moving_av_scenarios.py \
  --scenarios data/av2_mf_tiny/scenarios_rl.pt \
  --out data/av2_mf_tiny/scenarios_rl_moving_av_100.pt \
  --manifest data/av2_mf_tiny/scenarios_rl_moving_av_100.csv \
  --limit 100
```

## Outputs

Default outputs are written under `runs/`:

- `runs/ppo_final.pt`: PPO policy weights.
- `runs/ppo_final.json`: training configuration, training history, and eval history.
- `runs/ppo_final_report.png`: report plot generated from the history JSON.
- `runs/viz_one/*.png` or `*.gif`: one-scenario rollout visualizations.

## Notes

- `AV2_parallel.py` builds flat per-track data for TrajFlow.
- `AV2_scenarios.py` keeps full scenario structure for PPO.
- `scenario_env.py` is the main environment implementation: map constraints,
  PORA risk, goal reward, TTC/collision penalties, and episode termination.
- `dict_ppo.py` defines the heatmap-CNN + env-MLP actor-critic used by PPO.
