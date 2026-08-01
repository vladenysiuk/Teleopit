# Stage 0 — Baseline and Extension Points

Internal implementation note for `General-Climbing-G1`. Recorded during Stage 0
inspection; update when backend versions or extension patterns change.

## Dependency versions (pinned in `pyproject.toml` `[project.optional-dependencies].train`)

| Package | Pinned / resolved (Docker `teleopit-train:latest`, 2026-07-31) |
|---------|----------------------------------------------------------------|
| Python | 3.10.12 (Docker); host may differ |
| mjlab | 1.4.0 |
| rsl-rl-lib | 5.2.0 |
| mujoco | 3.8.1 |
| mujoco-warp | 3.8.1 |
| torch | 2.13.0+cu130 (Docker image); pin `torch>=2.7.0` in pyproject |
| CUDA (runtime) | 12.4.1 base image; torch built for cu130 |

Install training stack: `pip install -e ".[train]"`.

Canonical G1 asset: `assets/robots/unitree_g1/g1_29dof.xml` (via
`teleopit.runtime.assets.UNITREE_G1_XML`). Download with
`python scripts/setup/download_assets.py --only robots`.

## G1 control and timing (from `General-Tracking-G1`)

| Quantity | Value | Source |
|----------|-------|--------|
| Joint action dimension | 29 (`model.nu`) | canonical G1 XML |
| Free joint + joints (`nq`) | 36 | canonical G1 XML |
| Physics timestep | 0.005 s (200 Hz) | `tracking_env_cfg.sim.mujoco.timestep` |
| Decimation | 4 | `tracking_env_cfg.decimation` |
| Policy / env step | 0.02 s (50 Hz) | `decimation × sim_dt` |
| Action type | `JointPositionActionCfg` on entity `"robot"` | `tracking_env_cfg.actions["joint_pos"]` |
| Action scale | `G1_ACTION_SCALE` (16-group pattern from mjlab G1 cfg) | `env.py` overrides default 0.5 |

## Wrist bodies and tracking point offsets

Tracking task uses **`left_wrist_yaw_link`** and **`right_wrist_yaw_link`** as
end-effector bodies (terminations, wrist-position reward, self-collision exclude).

Hand / contact point offsets in robot body frame (from
`make_general_tracking_env_cfg` → `additional_wrist_pos` reward):

- `left_wrist_yaw_link`: `(0.18, -0.025, 0.0)`
- `right_wrist_yaw_link`: `(0.18, 0.025, 0.0)`

Full wrist kinematic chain in XML: `*_wrist_roll_link`, `*_wrist_pitch_link`,
`*_wrist_yaw_link` (left and right).

## Task registration pattern (tracking reference)

1. Constants in `train_mimic/tasks/tracking/config/constants.py`.
2. Env builder `make_general_tracking_env_cfg()` in `config/env.py` composes
   `make_tracking_env_cfg()` plus G1 robot spec, observation overrides, rewards.
3. PPO runner cfg in `config/rl.py`.
4. Custom runner class in `tasks/tracking/rl/runner.py`.
5. Side-effect registration in `config/registry.py`:

   ```python
   register_mjlab_task(
       task_id=GENERAL_TRACKING_TASK,
       env_cfg=make_general_tracking_env_cfg(),
       play_env_cfg=make_general_tracking_env_cfg(play=True),
       rl_cfg=make_general_tracking_ppo_runner_cfg(...),
       runner_cls=MotionTrackingOnPolicyRunner,
   )
   ```

6. `train_mimic/tasks/__init__.py` imports tracking registry only.

Climbing will mirror this under `train_mimic/tasks/climbing/` but **must not**
register until env cfg exists (Stage 0 skeleton only).

## Batched simulation access (mdp / managers)

- **Environment type**: `mjlab.envs.ManagerBasedRlEnv` with batched `num_envs`.
- **Observation / reward / termination terms** are plain functions
  `(env: ManagerBasedRlEnv, ...) -> torch.Tensor` shaped `(num_envs,)`.
- **Robot state**: resolve entity via `env.scene["robot"]` or
  `SceneEntityCfg("robot", ...)`; joint/body tensors are batched on `env.device`.
- **Built-in sensors**: `mdp.builtin_sensor(env, sensor_name="robot/imu_lin_vel")`
  reads mjlab sensor pipeline.
- **Custom commands**: subclass `CommandTerm` / `CommandTermCfg` (see
  `MotionCommand` in `tracking/mdp/commands.py`); access via
  `env.command_manager.get_term(name)`.
- **Actions**: `JointPositionActionCfg`; history via `mdp.last_action(env)`.
- **Contact**: `ContactSensorCfg` on `cfg.scene.sensors`; rewards read
  `env.scene.sensors[sensor_name]` (see `self_collision_cost`).
- **Domain randomization**: `EventTermCfg` with `mode="startup"` or `"interval"`,
  functions from `mjlab.envs.mdp.dr`.
- **Training entry**: `train_mimic/scripts/train.py` → `import_training_stack()`
  → `load_task_components()` → `ManagerBasedRlEnv` + custom runner.

Rewards and terminations for climbing should read **task state / sim tensors**,
not actor observations (per plan fixed decision #4).

## Proposed file map (Stages 1+)

```text
train_mimic/tasks/climbing/
├── config/
│   ├── constants.py      # done (Stage 0)
│   ├── registry.py       # stub (Stage 0); register in later stage
│   ├── env.py            # Stage 2+
│   ├── ladder.py         # Stage 1
│   └── rl.py             # Stage 6+
├── mdp/
│   ├── actions.py        # attach/detach
│   ├── events.py
│   ├── observations.py
│   ├── rewards.py
│   └── terminations.py
├── ladder/
│   ├── geometry.py
│   ├── generator.py
│   └── state.py
├── rl/
│   ├── ladder_encoders.py
│   └── climbing_model.py
└── tests/                # or repo-root tests/test_climbing_*.py

train_mimic/scripts/debug_climb.py   # grows per stage
```

## Baseline validation commands

```bash
# Import tracking registration (unchanged)
python -c "import train_mimic.tasks; print('ok')"

# Climbing skeleton only
python -c "import train_mimic.tasks.climbing; print('ok')"

# Narrow tests
pytest tests/test_task_registry.py tests/test_train_script.py tests/test_climbing_stage0.py -q
```
