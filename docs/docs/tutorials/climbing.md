---
sidebar_position: 6
---

# G1 Ladder Climbing (Training)

Train a Unitree G1 ladder-climbing policy in MuJoCo with `General-Climbing-G1`. This task is **separate** from `General-Tracking-G1` and does **not** share the 167D tracking ONNX contract.

:::info
For simulator internals see [Climbing Simulator Reference](../reference/climbing-simulator.md). For rewards, PPO, and curriculum see [Climbing RL Reference](../reference/climbing-rl.md).
:::

## Prerequisites

```bash
pip install -e ".[train]"
python scripts/setup/download_assets.py --only robots
python -c "import train_mimic.tasks; print('ok')"
```

Canonical robot model: `assets/robots/unitree_g1/g1_29dof.xml` (shared with tracking).

| Constant | Value |
|----------|-------|
| Task ID | `General-Climbing-G1` |
| Experiment name | `g1_general_climbing` |
| Physics rate | 200 Hz (`sim_dt=0.005`) |
| Policy rate | 50 Hz (`decimation=4`) |

## Quick validation path

Run these in order before a long training job:

```bash
# 1. Automated regression (CI-friendly)
python scripts/dev/run_climbing_regression.py --quick

# 2. Ladder scene viewer (macOS: mjpython)
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42

# 3. PPO infrastructure smoke
python train_mimic/scripts/train_climb.py --smoke

# 4. Easy learnability preset (short local run first)
python train_mimic/scripts/train_climb.py --easy --num_envs 64 --max_iterations 50
```

## Debug modes (`debug_climb.py`)

Single entry point for staged validation. Use `mjpython` on macOS for viewer modes.

| Mode | Purpose | Viewer |
|------|---------|--------|
| `scene` | Sampled ladder topology | Yes |
| `contacts` | Hand/rung contact identity | Yes |
| `latch` | Attach/detach scripted sequence | Yes |
| `observations` | Relative-rung values + debug lines | Yes |
| `rewards` | Per-term reward breakdown | Yes |
| `hold-pose` | Static four-contact gravity hold | Yes |
| `zero` | Zero-action MDP rollout | Headless |
| `random` | Random-action MDP rollout | Headless |
| `scripted-hold` | Stage 6 scripted hold agent | Optional |
| `scripted-mdp` | Full MDP scripted agent | Optional |
| `reset-stress` | Repeated randomized resets | Headless or viewer |
| `ppo-smoke` | Tiny PPO infrastructure run | Headless |
| `baseline-compare` | Zero vs random on easy env | Headless |
| `learnability` | Short train + baseline comparison | Headless |

Examples:

```bash
# Contact demo — move probe rung with Up/Down, switch hand with ,/ .
mjpython train_mimic/scripts/debug_climb.py --mode contacts --seed 42

# Reward term plots during scripted transitions
mjpython train_mimic/scripts/debug_climb.py --mode rewards --seed 42

# Headless reset stress
python train_mimic/scripts/debug_climb.py --mode reset-stress --headless --num-resets 200
```

## Training

### Smoke test (Stage 8 — infrastructure only)

Validates PPO plumbing, not climbing skill:

```bash
python train_mimic/scripts/train_climb.py --smoke
# 64 envs, 100 iterations, pinned easy ladder
```

### Easy learnability preset (Stage 9)

Fixed vertical 6-rung ladder, assisted hold IK reset, both hands attached at start, high cylinder friction, 15 s episodes:

```bash
python train_mimic/scripts/train_climb.py --easy
# Default: 256 envs/GPU, 800 iterations
```

### Custom training

```bash
python train_mimic/scripts/train_climb.py \
    --num_envs 4096 \
    --max_iterations 30000 \
    --seed 42
```

### Multi-GPU (single node)

`--num_envs` is **per GPU**. Total parallel envs = `num_envs × number of GPUs`.

**Explicit GPU IDs:**

```bash
python train_mimic/scripts/train_climb.py --easy \
    --gpu_ids 0 1 2 3 \
    --num_envs 128
# 512 total envs on 4 GPUs
```

**All visible GPUs:**

```bash
python train_mimic/scripts/train_climb.py --easy \
    --all_gpus \
    --num_envs 128
```

Notes:

- Do **not** pass `--device` in multi-GPU mode; each worker uses `cuda:{LOCAL_RANK}`.
- `--all_gpus` respects an existing `CUDA_VISIBLE_DEVICES` mask.
- Start with **64–128 envs/GPU** for climbing (heavier per env than tracking). Ramp after confirming no OOM.
- On 4× A100 (80 GB), **128–256 envs/GPU** (512–1024 total) is a reasonable target for `--easy`.

### Resume

```bash
python train_mimic/scripts/train_climb.py \
    --resume logs/rsl_rl/g1_general_climbing/<run>/model_400.pt \
    --max_iterations 800
```

`--max_iterations` adds iterations on top of the checkpoint iteration count.

### Logging

Default: TensorBoard under `logs/rsl_rl/g1_general_climbing/<timestamp>/`.

```bash
tensorboard --logdir logs/rsl_rl/g1_general_climbing
```

Optional: `--logger wandb` or `--logger swanlab`.

## Playback

```bash
# Easy checkpoint (matches train --easy MDP)
python train_mimic/scripts/play_climb.py \
    --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
    --easy --device cuda:0 --seed 42

# Headless multi-seed mp4 (8 reset samples → one video at 30% speed)
MUJOCO_GL=egl python train_mimic/scripts/play_climb.py \
    --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
    --easy --video --seed 42

# Browser viewer over SSH
python train_mimic/scripts/play_climb.py \
    --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
    --easy --viewer viser
```

`--video` concatenates `--video-clips` reseeds (default `8`, seeds `seed..seed+clips-1`) into a single Full HD (`1920x1080`) `play_climb.mp4` under `<checkpoint_dir>/videos/play/`. Each clip is one full episode unless you pass `--video-length`. Playback defaults to `--video-speed 0.3` (30% of realtime). Without `--easy`, each reseed also resamples ladder layout when curriculum randomization is enabled.

Smoke-ladder playback (Stage 8 checkpoints, no curriculum reset):

```bash
python train_mimic/scripts/play_climb.py \
    --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_100.pt \
    --smoke-ladder
```

## Automated tests

Full regression:

```bash
python scripts/dev/run_climbing_regression.py
python scripts/dev/run_climbing_regression.py --quick   # skip long PPO/GPU tests
```

Direct pytest:

```bash
pytest tests/test_climbing_stage*.py tests/test_climbing_regression.py -v
```

## Package layout

Developer reference (staged gate notes live under `train_mimic/tasks/climbing/STAGE*.md`):

```text
train_mimic/tasks/climbing/
├── config/          # env, ladder, latch, rewards, curriculum, rl, registry
├── ladder/          # geometry, generator, state, contacts, latch, relative_rungs
├── mdp/             # actions, observations, rewards, terminations, events, metrics
├── rl/              # ClimbingModel, ladder encoders, runner
└── debug_*.py       # mode-specific helpers

train_mimic/scripts/
├── debug_climb.py   # unified debug entry
├── train_climb.py   # PPO training
└── play_climb.py    # checkpoint playback
```

## Known limitations

- **No production depth camera.** Privileged relative-rung geometry is the v1 perception path.
- **Simulation only.** Point-hand spheres and `connect` latches are MuJoCo abstractions, not a sim-to-real deployment claim. Training latches default to a **500 N** overload break and softened `solref` so an attached hand cannot act as an infinite-strength pivot.
- **mujoco_warp sensor patch.** `train_mimic.warp_patches` fixes a `_frame_axis` UNKNOWN-branch codegen bug (`undefined symbol: xmat`) in mujoco_warp 3.8.x before env construction. Climbing entry points apply it automatically.
- **Fixed compiled rung count.** Ladder topology is compiled once; poses and active masks vary at reset.
- **Curriculum axes exist but easy preset keeps randomization off.** Expand one axis at a time after learnability is confirmed.

## Explicit non-goals

- RGB-D / depth perception for production
- Sim-to-real climbing deployment
- Foot attachment constraints
- Per-episode MJCF recompilation
- Preserving the 167D tracking ONNX observation contract
