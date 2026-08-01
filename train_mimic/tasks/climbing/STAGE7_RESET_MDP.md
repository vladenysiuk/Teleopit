# Stage 7 — Reset robustness, scripted agents, and end-to-end MDP checks

## Scope

Validate the complete `General-Climbing-G1` MDP without PPO:

- repeated randomized resets with genuine ladder resampling (active rung count,
  spacings, pose/tilt, inactive masks/parking);
- post-reset zero-action steps (3–5) to catch unstable penetrations and stale constraints;
- explicit verification of all reset-sensitive state (latch, contacts, reward memory,
  previous actions, 10-frame observation history, terminations, metrics);
- zero-action and random-action rollouts (mjlab-compatible dummy agents);
- **separate** Stage 6 assisted-hold regression vs full-MDP scripted test;
- reward/termination/metric logging through whole episodes;
- simulation capacity checks (`nconmax`, `njmax`, active equalities);
- CPU batches 1/4/16 in CI; local GPU batch 64 before first PPO run;
- fixed-seed reproducibility on pinned geometry.

Stage 7 does **not** start PPO training (Stage 8).

## Files

| Path | Role |
|------|------|
| `train_mimic/tasks/climbing/debug_mdp.py` | Agents, rollouts, reset stress, capacity snapshots, benchmark report |
| `train_mimic/tasks/climbing/reset_state.py` | Reset-state capture, pollution, and verification helpers |
| `train_mimic/tasks/climbing/config/env.py` | `make_climbing_mdp_env_cfg()` |
| `train_mimic/scripts/debug_climb.py` | `--mode zero\|random\|scripted-hold\|scripted-mdp\|reset-stress` |
| `tests/test_climbing_stage7.py` | Automated reset/agent/batch/reproducibility suite |

## Architectural choices

1. **`make_climbing_mdp_env_cfg`** wraps the full registered env with configurable episode
   length. Reset-stress uses `randomized_ladder_cfg()`; rollouts/reproducibility use
   `pinned_ladder_cfg()`.

2. **Dummy agents** mirror mjlab `play --agent zero|random`.

3. **Split scripted agents**
   - `AssistedHoldScriptedAgent` — Stage 6 hold-pose env, stiff PD hold only (no root nudge).
   - `FullMdpScriptedAgent` — registered MDP, latch-only actions after assisted init.
   - Fall / invalid-attach root nudges are **test-side fault injection**
     (`make_mdp_fault_injection_hook`), not agent actions.

4. **`stress_random_resets`** pollutes all envs, resets, asserts clean state, then steps
   3 zero-action frames per reset. Randomized stress requires `ladder_variants > 1`.

5. **Observation history** — mjlab pushes once on reset obs compute; post-reset checks
   use baseline `HISTORY_PUSHES_AFTER_RESET = 1`.

6. **Capacity guard** — global `nacon_total < nconmax × num_envs`; per-env `max_nefc < njmax`.

## Commands

```bash
pytest tests/test_climbing_stage7.py -v

python train_mimic/scripts/debug_climb.py --mode zero --headless --seed 42
python train_mimic/scripts/debug_climb.py --mode random --headless --seed 42 --num-envs 4
python train_mimic/scripts/debug_climb.py --mode scripted-hold --headless --seed 42
python train_mimic/scripts/debug_climb.py --mode scripted-mdp --headless --seed 42
python train_mimic/scripts/debug_climb.py --mode reset-stress --headless --seed 42 --num-resets 300
```

GPU capacity (local, before Stage 8 PPO):

```bash
pytest tests/test_climbing_stage7.py::test_gpu_batch_64_capacity -v
```

## Observed rollout results (seed 42, CPU, pinned ladder, 120 steps unless noted)

### Zero agent

| Field | Value |
|-------|-------|
| Steps | 120 |
| Completed episodes | 0 |
| Reward sum | 0.1618 |
| Terminations | none (episode still active at 2 s horizon) |
| Reward terms | upward_progress=0.2416, time_penalty=−0.0300, effort=−0.0498 |
| Max valid attachments | 0 |
| Max invalid latch | 0 |
| Max nacon / max nefc | 58 / 232 |
| Min ncon / njmax headroom | 326 / 968 |
| Nonfinite obs/sim steps | 0 / 0 |

### Random agent

| Field | Value |
|-------|-------|
| Steps | 120 |
| Completed episodes | 0 |
| Reward sum | −14.8071 |
| Terminations | none |
| Reward terms | upward_progress=0.6020, invalid_latch=−15.0, action_rate=−0.1229, effort=−0.2562 |
| Max valid attachments | 0 |
| Max invalid latch | 30 |
| Max nacon / max nefc | 56 / 224 |
| Min ncon / njmax headroom | 328 / 976 |
| Nonfinite obs/sim steps | 0 / 0 |

### Full-MDP scripted (`scripted-mdp`, 88 steps, fault injection enabled)

| Field | Value |
|-------|-------|
| Steps | 88 |
| Completed episodes | 1 |
| Episode length | 78 (mean/min/max) |
| Reward sum | 6.4606 |
| Terminations | terminated=1 |
| Reward terms | upward_progress=7.7784, invalid_latch=−0.5, effort=−0.7955, time_penalty=−0.0220 |
| Max valid attachments | 0 |
| Max invalid latch | 1 |
| Max equalities per env | 2 |
| Max nacon / max nefc | 56 / 224 |
| Min ncon / njmax headroom | 328 / 976 |
| Nonfinite obs/sim steps | 0 / 0 |

### Stage 6 assisted-hold regression (hold-pose env, 2.0 s)

| Field | Value |
|-------|-------|
| Failure class | ok |
| Max pelvis drop | 0.11146235466003418 m |
| Max foot slip | 0.07081150263547897 m |
| Mean foot contact | 0.6549479166666666 |
| Any foot contact | 1.0 |
| Max hand attach error | 0.004356163553893566 m |
| Max nacon / max nefc | 24 / 102 |

### Randomized reset stress (50 resets × 3 post-reset zero-action steps)

| Field | Value |
|-------|-------|
| Resets | 50 |
| Ladder variants | 50 |
| Stale state failures | 0 |
| Nonfinite steps | 0 |
| Max nacon / max nefc | 60 / 240 |
| Min ncon / njmax headroom | 324 / 960 |
| OK | true |

300-reset CI test: `test_randomized_300_resets_with_post_steps` (same helper, passes).

### Reproducibility (pinned ladder, seed 42, 5-step mixed zero/random)

| Field | Value |
|-------|-------|
| Signature match across two runs | true |
| Signatures recorded | 5 |

## Exit criteria

| Criterion | Evidence |
|-----------|----------|
| 300 randomized resets + 3 post-reset steps | `test_randomized_300_resets_with_post_steps` |
| Pinned reproducibility separate from randomized stress | `test_pinned_geometry_reset_reproducibility` |
| All reset-sensitive stores verified | `reset_state.py` + reset tests |
| Observation history cleared (no stale latch frames) | `test_observation_history_cleared_after_reset` |
| Subset reset isolation | `test_subset_reset_preserves_unselected_state` |
| Zero/random/full-MDP scripted agents | rollout tests + debug modes |
| Stage 6 hold regression separate | `test_assisted_hold_scripted_regression`, `test_scripted_hold_reproduces_stage6_thresholds` |
| CPU batch 1/4/16 | parametrized capacity test |
| GPU batch 64 (local) | `test_gpu_batch_64_capacity` (skipped without CUDA) |
| Tracking task untouched | registry test |

## Known limitations

- GPU batch 64/256 not run in CI; execute locally before Stage 8 PPO smoke.
- Full-MDP scripted fall depends on test-side root nudge + fall termination wiring.
- Detailed force/torque diagnostics deferred to Stage 8/9.
- PPO smoke test remains Stage 8.

## Next stage

Stage 8 — tiny PPO smoke test and reward-scale diagnosis.
