---
sidebar_position: 8
---

# Climbing RL Reference

Implementation reference for the **RL-side** of `General-Climbing-G1`: reward terms, terminations, metrics, event memory, PPO/model configuration, curriculum, training scripts, and learnability harness.

Simulator internals (ladder, contacts, latch, observations) are in [Climbing Simulator Reference](./climbing-simulator.md). Usage commands are in the [Climbing Tutorial](../tutorials/climbing.md).

## Architecture overview

```text
ManagerBasedRlEnv (mjlab)
        │
        ├── reward_manager  ◄── mdp/rewards.py      (reads ClimbRewardState, contacts, latch)
        ├── termination_manager ◄── mdp/terminations.py
        ├── metrics_manager ◄── mdp/metrics.py
        │
        ▼
ClimbingModel (actor + critic)
        ├── Conv1dEncoder × proprio history groups
        ├── LadderEncoder × relative rungs [B,K,7]
        └── MLP head (2048,1024,512,256,128)
        │
        ▼
ClimbingOnPolicyRunner ──► rsl_rl PPO
        ├── per-term RewardTerm/* logging
        ├── Policy/latch_action_* logging
        └── ObsNorm/* logging
```

**Fixed design decision (RL):** reward and termination functions consume **simulator task state** (`ClimbRewardState`, `ClimbContactState`, `ClimbLatchState`, body tensors) — **not** actor observation tensors. Changing the perception provider later must not change reward semantics.

## Package map (RL)

| Path | Role |
|------|------|
| `config/rewards.py` | `ClimbingRewardConfig`, `ClimbingTerminationConfig`, wiring |
| `config/rl.py` | PPO runner defaults, `ClimbingModel` class path |
| `config/easy.py` | Stage 9 easy env + PPO presets |
| `config/smoke.py` | Stage 8 smoke presets |
| `config/curriculum.py` | `CurriculumConfig`, axis toggles |
| `config/registry.py` | Task registration |
| `mdp/rewards.py` | Individual reward term functions → `[B]` |
| `mdp/terminations.py` | Success, fall, timeout, NaN, latch overload |
| `mdp/metrics.py` | Episode metrics (max height, attachments, etc.) |
| `ladder/reward_state.py` | `ClimbRewardState` — anti-farming event memory |
| `curriculum.py` | Reset-time pose seeding, physics sampling |
| `rl/climbing_model.py` | Fused proprio + history + ladder latent model |
| `rl/ladder_encoders.py` | Vector + depth encoder interfaces |
| `rl/runner.py` | `ClimbingOnPolicyRunner` diagnostics |
| `ppo_smoke.py` | Reusable Stage 8 smoke harness |
| `learnability.py` | Zero/random baselines + short-train comparison |
| `scripts/train_climb.py` | Training CLI |
| `scripts/play_climb.py` | Playback CLI |

## Action space

| Index | Term | Dim | Description |
|------:|------|----:|-------------|
| 0 | `joint_pos` | 29 | Joint position offsets (`G1_ACTION_SCALE`, default standing offset) |
| 1 | `latch` | 2 | Left/right attach-detach commands |

Target joint positions: `clip(action, -10, 10) × scale + default_dof_pos`.

## Reward terms

All weights live in `ClimbingRewardConfig` (`config/rewards.py`). Functions in `mdp/rewards.py` return shape `[B]`.

| Term | Default weight | Semantics |
|------|---------------:|-----------|
| `upward_progress` | +2.0 | Head (`d435i_link`) **ladder-relative** height delta (not pelvis/hand; pelvis pays for inverted salting) |
| `new_higher_attachment` | +10.0 | One-off event per new higher rung attachment |
| `success` | +50.0 | Head above top rung − clearance with min attached hands |
| `time_penalty` | −0.05 × dt | Physical-time cost while not successful |
| `action_rate` | −0.01 | L2 on action delta |
| `effort` | −1.0e-4 | Joint torque L2 |
| `invalid_latch` | −0.5 | Attach request without valid contact |

Optional `latch_overload` penalty/termination exists but defaults **off**.

### Event memory (`ladder/reward_state.py`)

`ClimbRewardState` prevents attachment-reward farming:

- Maintaining an attachment earns **no** repeated bonus.
- Detach/reattach to the same or lower rung does not re-pay.
- Memory resets with the environment on the `reset_climb_rewards` event.

Success predicate (configurable via `ClimbingRewardConfig`):

- Head (`d435i_link`) ladder-relative height ≥ top rung − `success_pelvis_clearance_below_top_l` (default 0.15 m).
- At least `success_min_attached_hands` (default 2) with attachment near ladder top.

## Terminations

| Term | Default | Condition |
|------|---------|-----------|
| `success` | on | Success predicate (above) |
| `fall` | on | Pelvis too low or too far from ladder (XY) |
| `time_out` | on | Episode length exceeded |
| `nan` | on | Non-finite state |
| `latch_overload` | off | Constraint force threshold |

## Episode metrics

Logged via `metrics_manager` (`mdp/metrics.py`):

| Metric | Description |
|--------|-------------|
| `max_head_height_l` | Running max head / progress-body ladder-relative height (`d435i_link` by default) |
| `head_height_l` | Current head / progress-body ladder-relative height |
| `valid_higher_attachments` | Count of valid higher attachment events |
| `invalid_latch_count` | Invalid attach requests |
| `success` | Episode success flag |
| `hand_contact_fraction` | Mean hand contact indicator |
| `time_to_success` | Steps to first success, averaged over successful episodes only (per-env sentinel remains −1 until success; batches with no successes omit the metric) |
| `torque_saturation_fraction` | Mean torque saturation indicator |

## PPO configuration

**Runner:** `ClimbingOnPolicyRunner` extends tracking runner with:

- `RewardTerm/{name}` scalars each iteration
- `Policy/latch_action_{mean,std,abs_mean}`
- `ObsNorm/*` from actor/ladder normalizers
- `Episode_Metrics/time_to_success` averaged over successful episodes only
- `Episode_Termination/*` summed per iteration (mjlab emits per-reset counts; averaging them produced fractional totals)

**Defaults** (`config/rl.py`):

| Setting | Value |
|---------|-------|
| Model | `ClimbingModel` |
| Hidden dims | 2048, 1024, 512, 256, 128 |
| `num_steps_per_env` | 24 |
| `max_iterations` | 30,000 |
| `save_interval` | 2000 |
| History length | 10 (proprio) |
| Ladder latent | 64-D |

**Observation groups fed to policy:**

| Set | Groups |
|-----|--------|
| Actor | `actor_proprio`, `actor_proprio_history`, `actor_ladder` |
| Critic | `critic_proprio`, `critic_proprio_history`, `critic_ladder`, `critic_privileged` |

**Model fusion** (`rl/climbing_model.py`):

1. Conv1d temporal encoders on history groups.
2. `LadderEncoder` on `[B, K=6, 7]` relative rung tensor (valid bit excluded from normalization).
3. Concatenate latents → shared MLP actor/critic heads.

Dummy depth encoder path exists for config-only tests; production env rejects `mode="depth"`.

## Training presets

### Smoke (`config/smoke.py` + `--smoke`)

| Setting | Value |
|---------|-------|
| Envs | 64 |
| Iterations | 100 |
| Ladder | Pinned easy (6 rungs, fixed spacing) |
| Episode | 10 s |
| Purpose | PPO infrastructure validation |

### Easy learnability (`config/easy.py` + `--easy`)

| Setting | Value |
|---------|-------|
| Envs | 256 / GPU |
| Iterations | 800 |
| Episode | 15 s |
| Ladder | Fixed vertical 6-rung |
| Reset | Assisted hold IK + both hands attached (`initial_hand_attach_prob=1.0`) |
| Friction | High cylinder + foot friction bump |
| Randomization | All curriculum axes **off** |

CI subset: 4 envs, 12 iterations (`learnability` debug mode).

## Curriculum

`CurriculumConfig` (`config/curriculum.py`) exposes axes as `randomize_*` flags — no hard-coded stage switches.

**Expansion order (recommended):**

1. `initial_hand_attach_prob < 1.0`
2. `randomize_spacing=True`
3. `randomize_rung_count=True`
4. `randomize_ladder_pose=True`
5. `randomize_rung_physics=True`
6. `randomize_initial_pose=True`
7. `randomize_latch_capture=True`

Enable one axis at a time after the fixed-ladder task shows a learning signal.

## Multi-GPU training

Uses shared helpers in `train_mimic/distributed_launch.py`:

```bash
# Explicit IDs
python train_mimic/scripts/train_climb.py --easy --gpu_ids 0 1 2 3 --num_envs 128

# All visible GPUs
python train_mimic/scripts/train_climb.py --easy --all_gpus --num_envs 128
```

- `--num_envs` is per GPU; rsl_rl syncs gradients via `torchrun` + NCCL.
- Do not pass `--device` under multi-GPU.
- Seeds offset per rank: `seed + rank × 100003`.

**Suggested env counts (4× A100 80 GB, `--easy`):** start 128/GPU (512 total), ramp toward 256/GPU (1024 total) after profiling.

## Learnability harness

`learnability.py` + `debug_climb.py --mode learnability`:

1. Roll out **zero** and **random** baselines on easy env.
2. Train short PPO run (CI: 12 iters).
3. Compare max head height and valid higher attachments.

`learning_signal` is true when trained policy beats either baseline on those metrics.

Owner validation: multi-seed videos, attachment sequences, foot contact/slip — not a single cherry-picked episode.

## Playback

`play_climb.py` loads checkpoint into matching MDP:

| Flag | MDP |
|------|-----|
| `--easy` | Same as `train_climb.py --easy` |
| `--smoke-ladder` | Pinned ladder, no curriculum reset |

Supports native viewer, Viser (SSH), and headless `--video` (EGL). Video mode concatenates multiple reseeds into one Full HD mp4 at 30% playback speed by default (`--video-clips 8`, `--video-speed 0.3`, `1920x1080`).

## Smoke / reward-scale diagnosis

`ppo_smoke.py` flags reward-scale pathology:

- Penalty term &gt; 10× combined progress terms.
- `invalid_latch` &gt; 75% of |total reward|.

Weights are **not** auto-tuned — adjust in `ClimbingRewardConfig` after contact learning, not before.

## Task registration

```python
# train_mimic/tasks/climbing/config/registry.py
register_mjlab_task(
    task_id="General-Climbing-G1",
    env_cfg=make_general_climbing_env_cfg(),
    play_env_cfg=make_general_climbing_env_cfg(play=True),
    rl_cfg=make_general_climbing_ppo_runner_cfg(),
    runner_cls=ClimbingOnPolicyRunner,
)
```

Imported from `train_mimic/tasks/__init__.py`. Tracking remains `DEFAULT_TASK`.

## Testing

| Suite | Command |
|-------|---------|
| Full regression | `python scripts/dev/run_climbing_regression.py` |
| Quick CI | `python scripts/dev/run_climbing_regression.py --quick` |
| Stage 5 rewards | `pytest tests/test_climbing_stage5.py -v` |
| Stage 8 PPO smoke | `pytest tests/test_climbing_stage8.py -v` |
| Stage 9 learnability | `pytest tests/test_climbing_stage9.py -v` |

## Known RL limitations

- No ONNX export contract for climbing yet (separate from tracking 167D dual-input).
- `invalid_latch` can dominate early random policies; easy preset uses guaranteed initial attachment before weight tuning.
- CI learnability run is short — validates plumbing, not research-grade learning.
- Broad domain randomization is intentionally deferred until fixed-task learnability is demonstrated.

## Related stage gate notes

| Stage | Doc |
|------:|-----|
| 5 | `STAGE5_REWARDS.md` |
| 8 | `STAGE8_PPO_SMOKE.md` |
| 9 | `STAGE9_LEARNABILITY.md` |
| 10 | `STAGE10_HANDOFF.md` |
