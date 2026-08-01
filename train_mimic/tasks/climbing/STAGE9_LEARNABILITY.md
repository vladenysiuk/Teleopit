# Stage 9 — First learnability experiment and curriculum hooks

## Scope

Add configuration-only curriculum controls and the first **easy** climbing
experiment preset. Stage 9 validates that a measurable learning signal can exist
on the fixed-ladder task before enabling broader randomization.

Stage 9 does **not** enable full domain randomization or claim sim-to-real
climbing feasibility.

## Files

| Path | Role |
|------|------|
| `train_mimic/tasks/climbing/config/curriculum.py` | `CurriculumConfig`, `easy_curriculum_cfg()`, ladder/latch builders |
| `train_mimic/tasks/climbing/config/easy.py` | Easy env + PPO presets (256 envs, 800 iters owner default) |
| `train_mimic/tasks/climbing/curriculum.py` | Reset-time pose seeding, physics/capture sampling |
| `train_mimic/tasks/climbing/mdp/events.py` | Curriculum reset event wrappers |
| `train_mimic/tasks/climbing/debug_hold_pose.py` | Batched noisy hold pose + probabilistic latch seeding |
| `train_mimic/tasks/climbing/learnability.py` | Zero/random baselines, short train, metric comparison |
| `train_mimic/scripts/train_climb.py` | `--easy` owner training preset |
| `train_mimic/scripts/debug_climb.py` | `--mode baseline-compare`, `--mode learnability` |
| `tests/test_climbing_stage9.py` | Curriculum + baseline + CI learnability harness |

## Architectural choices

1. **Curriculum is configuration, not code branches.** Each axis is a
   `randomize_*` flag plus numeric ranges on `CurriculumConfig`. The Stage 9
   easy preset keeps all axes disabled.

2. **Easy experiment MDP:**
   - fixed vertical 6-rung ladder (pinned spacing/distance/yaw/tilt);
   - high cylinder friction + optional foot friction bump (Stage 6 allowed);
   - assisted hold IK reset with both hands force-attached (`initial_hand_attach_prob=1.0`);
   - modest episode length (15 s).

3. **Reset order:** latch deactivate → contacts clear → ladder resample → reward
   memory clear → curriculum physics → curriculum pose (IK + latch seed).

4. **Latch capture radius** is stored per-env on `HandLatchState.capture_radius`
   so a future curriculum axis can vary eligibility without changing attach logic.

5. **Learnability harness** compares zero/random rollouts vs a short trained
   checkpoint on the **same easy env + seed**. `learning_signal` is true when
   pelvis height or valid higher attachments improve vs either baseline.

6. **Broad randomization remains off.** Use `with_curriculum_axis()` to enable
   one axis at a time after the owner approves expansion.

## Curriculum expansion order (owner gate)

1. `initial_hand_attach_prob < 1.0`
2. `randomize_spacing=True`
3. `randomize_rung_count=True`
4. `randomize_ladder_pose=True` (+ per-env hold IK)
5. `randomize_rung_physics=True`
6. `randomize_initial_pose=True`
7. `randomize_latch_capture=True`

## Commands

```bash
# Fast CI
pytest tests/test_climbing_stage9.py -v

# Zero vs random baselines on easy env
python train_mimic/scripts/debug_climb.py --mode baseline-compare --seed 42

# CI learnability harness (4 envs, 12 PPO iters)
python train_mimic/scripts/debug_climb.py --mode learnability --seed 42

# Owner easy training preset (256 envs, 800 iterations)
python train_mimic/scripts/train_climb.py --easy
```

## Pre-run success evidence (owner)

Compare trained vs zero/random on several fixed seeds:

| Metric | Expected direction |
|--------|-------------------|
| max pelvis ladder-relative height | trained > baselines |
| valid higher attachments | trained ≥ baselines |
| invalid latch rate | trained ≤ baselines |
| torque saturation | no large increase vs baselines |

Inspect multi-seed videos, attachment sequences, foot contact/slip, and
time-to-progress — not a single cherry-picked episode.

## Exit criteria

| Criterion | Evidence |
|-----------|----------|
| Curriculum controls exist as config | `CurriculumConfig` + axis toggles |
| Easy preset fixed-ladder + assisted start | `easy_curriculum_cfg`, easy env tests |
| Baseline comparison runs | `test_baseline_comparison_runs_on_easy_env` |
| Short train + checkpoint playback | `test_learnability_experiment_completes` |
| Learning signal metric available | `LearnabilityComparison.learning_signal` |
| Broad randomization not enabled | easy preset flags all false |

## Known limitations

- Per-env hold IK on randomized ladders is supported but expensive; easy preset
  caches one IK solution for the pinned ladder.
- Rung **radius** randomization requires compile-time `LadderConfig.rung_radius`;
  runtime curriculum varies sliding friction only until radius recompile is added.
- CI learnability run is short (12 iterations) — it validates plumbing, not
  research-grade learning. Owner runs use 800+ iterations on GPU.
- Stage 9 does not tune reward weights; address `invalid_latch` dominance via
  curriculum (guaranteed attach) before weight changes.

## Next stage

Stage 10 — documentation, regression suite, and handoff.
