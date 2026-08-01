# Stage 8 — Tiny PPO smoke test and reward-scale diagnosis

## Scope

Validate end-to-end PPO integration for `General-Climbing-G1` without claiming
climbing learnability:

- climbing-specific PPO/model configuration via the existing mjlab + rsl_rl path;
- short smoke run (64 envs, ~100 iterations, pinned easy ladder);
- per-term reward, latch-action, and observation-normalization logging;
- checkpoint save + playback;
- automated reward-scale diagnosis (flags `invalid_latch` dominance).

Stage 8 does **not** start the Stage 9 learnability experiment.

## Files

| Path | Role |
|------|------|
| `train_mimic/tasks/climbing/config/smoke.py` | Smoke env + PPO presets |
| `train_mimic/tasks/climbing/config/rl.py` | Production PPO defaults (`max_iterations=30000`) |
| `train_mimic/tasks/climbing/rl/runner.py` | `ClimbingOnPolicyRunner` diagnostics |
| `train_mimic/tasks/climbing/ppo_smoke.py` | Reusable smoke harness + reward-scale diagnosis |
| `train_mimic/scripts/train_climb.py` | Climbing training entry point (`--smoke`) |
| `train_mimic/scripts/play_climb.py` | Checkpoint playback (no motion dataset) |
| `train_mimic/scripts/debug_climb.py` | `--mode ppo-smoke` |
| `tests/test_climbing_stage8.py` | Automated smoke + registry checks |

## Architectural choices

1. **`ClimbingOnPolicyRunner`** subclasses `MotionTrackingOnPolicyRunner` and logs:
   - `RewardTerm/{name}` from the env reward manager each iteration;
   - `Policy/latch_action_{mean,std,abs_mean}` for the last two action dims;
   - `ObsNorm/*` from actor/ladder empirical normalizers.

2. **Smoke env** uses `pinned_ladder_cfg()` (6 active rungs, fixed spacing/distance,
   zero yaw/tilt), `decimation=4`, `episode_length_s=10`, no extra domain
   randomization events.

3. **Task registration** wires `runner_cls=ClimbingOnPolicyRunner`. Production RL
   cfg uses `max_iterations=30000`; smoke cfg overrides to 100 iterations /
   `save_interval=50`.

4. **`train_mimic.app.SUPPORTED_TASKS`** now includes `General-Climbing-G1` so
   shared loaders accept the climbing task id. Tracking remains the default task.

5. **Reward-scale diagnosis** flags penalty terms whose average magnitude exceeds
   10× combined progress terms, and `invalid_latch` when it exceeds 75% of |reward|.
   Weights are **not** changed automatically — Stage 7 random-agent data already
   showed `invalid_latch` can dominate before contact learning; tune in Stage 9.

## Commands

```bash
# Fast CI smoke (4 envs, 3 iterations)
pytest tests/test_climbing_stage8.py -v

# Owner smoke preset (64 envs, 100 iterations)
python train_mimic/scripts/train_climb.py --smoke

# Equivalent debug entry point
python train_mimic/scripts/debug_climb.py --mode ppo-smoke --seed 42

# Playback last smoke checkpoint
python train_mimic/scripts/play_climb.py \
  --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_100.pt \
  --smoke-ladder
```

## TensorBoard scalars (smoke run)

| Group | Keys |
|-------|------|
| Train | `mean_reward`, `mean_episode_length`, `nonfinite_reward_steps` |
| RewardTerm | `upward_progress`, `new_higher_attachment`, `success`, `time_penalty`, `action_rate`, `effort`, `invalid_latch` |
| Policy | `mean_std`, `latch_action_mean`, `latch_action_std`, `latch_action_abs_mean` |
| ObsNorm | `mean_abs`, `std_mean`, `ladder_{group}_mean_abs` |
| Loss | PPO losses, `learning_rate` |
| Episode | climbing metrics from mjlab (`max_pelvis_height_l`, `invalid_latch_count`, …) |
| Perf | `total_fps`, `collection_time`, `learning_time` |

## Exit criteria

| Criterion | Evidence |
|-----------|----------|
| Training starts and checkpoints | `test_ppo_smoke_run_is_finite_and_checkpoints` |
| Finite losses/obs/rewards | smoke harness + runner NaN guards |
| Multiple episodes/resets | `completed_episodes > 0` in smoke report |
| Reward variance logged | smoke report |
| Latch actions trained/logged | `Policy/latch_action_*` + smoke report |
| Reward decomposition sensible | TensorBoard + `diagnose_reward_scale` |
| Playback loads checkpoint | `playback_ok` in smoke report |
| Tracking task untouched | existing registry/tests |

## Known limitations

- Smoke run does not prove climbing success; judge only infrastructure stability.
- Early policies may spam invalid latch attach → large `invalid_latch` penalty
  (documented; address via curriculum/weights in Stage 9).
- No conservative IK initial pose in smoke env yet — default G1 spawn near the
  pinned ladder is sufficient for plumbing validation.
- GPU batch 64 smoke should be run locally before the first long experiment.

## Next stage

Stage 9 — first learnability experiment and curriculum hooks.
