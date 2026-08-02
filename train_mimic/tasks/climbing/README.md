# General-Climbing-G1

MuJoCo ladder-climbing RL task for the Unitree G1 (Stages 0–10).

## Documentation

**User-facing docs (recommended):**

| Doc | Content |
|-----|---------|
| [Climbing Tutorial](../../../docs/docs/tutorials/climbing.md) | Installation, debug modes, training, playback, multi-GPU |
| [Simulator Reference](../../../docs/docs/reference/climbing-simulator.md) | Ladder, contacts, latch, observations, hold pose |
| [RL Reference](../../../docs/docs/reference/climbing-rl.md) | Rewards, PPO, curriculum, learnability |

Online: [BotRunner64.github.io/Teleopit/tutorials/climbing](https://BotRunner64.github.io/Teleopit/tutorials/climbing)

**Stage gate notes (implementation history):**

| Stage | Doc |
|------:|-----|
| 0–10 | `STAGE0_BASELINE.md` … `STAGE10_HANDOFF.md` |

## Quick commands

```bash
# Regression
python scripts/dev/run_climbing_regression.py --quick

# Scene viewer (macOS: mjpython)
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42

# Train easy preset
python train_mimic/scripts/train_climb.py --easy

# Multi-GPU (all visible GPUs, 128 envs/GPU)
python train_mimic/scripts/train_climb.py --easy --all_gpus --num_envs 128
```

Implementation plan: [`teleopit_climbing_task_plan.md`](../../../teleopit_climbing_task_plan.md)
