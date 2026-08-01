# Stage 5 — Reward Terms, Event Memory, Metrics, and Terminations

Internal handoff note for `General-Climbing-G1`. Update when reward weights,
success predicates, metrics, or debug tooling change.

## Scope

Stage 5 delivers:

- independently testable reward terms reading simulator task state;
- configurable weights/thresholds (`ClimbingRewardConfig`, `ClimbingTerminationConfig`);
- per-episode reward event memory (`ClimbRewardState`);
- terminations: success, fall, timeout, optional NaN/overload;
- episode metrics via mjlab `MetricsTermCfg` (no zero-weight reward hooks);
- `debug_climb.py --mode rewards`.

Stage 5 does **not** add static IK hold validation or PPO smoke training
(Stage 6+ / Stage 8).

## Reward terms and `dt` scaling

mjlab computes `r_t = dt * Σ_i w_i f_i` when `scale_by_dt=True`. Therefore:

| Kind | Raw `f` returned by term | Step contribution |
|------|--------------------------|-------------------|
| novel progress | `Δh_novel / dt` | `w Δh_novel` |
| sparse events | `1_event / dt` | `w` |
| time mask | `1` while not success | `w dt` |
| dense regularizers | mjlab L2 rates | `w · rate · dt` |

| Term | Raw output | Default weight | Notes |
|------|------------|----------------|-------|
| `upward_progress` | novel pelvis Δh / dt | `+10.0` | telescopes to `w(max h − h0)`; no oscillation farming |
| `new_higher_attachment` | `1_event / dt` | `+2.0` | height eps `1e-4`; one bonus if both hands attach same height |
| `success` | `1_event / dt` | `+20.0` | first qualifying success step only |
| `time_penalty` | `1` while not yet successful | `-0.05` | physical-time via mjlab `dt` |
| `action_rate` | mjlab `action_rate_l2` | `-0.01` | full 31-D action vector |
| `effort` | mjlab `joint_torques_l2` | `-1e-4` | actuator force L2 |
| `invalid_latch` | `1_attempt / dt` | `-0.5` | **one penalty per attach attempt** (edge-triggered) |
| `latch_overload` | optional | disabled | enabled only when weight set |

There is **no** `_step_recorder` reward term. mjlab skips `weight == 0` terms, so
counters must not depend on a zero-weight hook.

### Event memory (`ClimbRewardState`)

Stored in `env.extras["climb_reward_state"]`. Only quantities required by rewards:

| Field | Purpose |
|-------|---------|
| `max_rewarded_pelvis_height_l` | novel progress baseline; init = reset pelvis height |
| `max_rewarded_attachment_height_l` | highest paid attach height; `-inf` after reset |
| `success_rewarded` | suppresses repeated success bonus |
| `valid_higher_attachment_count` | metric counter updated by attach reward |
| `invalid_latch_count` | metric counter updated by invalid reward |
| `invalid_latch_armed` | arms one penalty per attach-command attempt |
| `time_to_success_steps` | set from `episode_length_buf` on first success |

Reset order (after Stage 5):

1. deactivate latch equalities;
2. clear contact state;
3. resample ladder;
4. reset reward memory anchored to the new pelvis height.

## Success and fall predicates

Shared success predicate (`climbing_success_predicate`):

```text
pelvis_height_l >= top_active_rung_height_l - success_pelvis_clearance_below_top_l
AND attached_hands >= success_min_attached_hands
AND at least one attached hand is on the top rung
    OR within success_top_attach_margin_l of the top rung height
```

Defaults:

- `success_pelvis_clearance_below_top_l = 0.15` m;
- `success_top_attach_margin_l = 0.05` m;
- `success_min_attached_hands = 1`.

This keeps success attainable on short (4-rung) and tall (10-rung) sampled
ladders and rejects “high pelvis + low rung latch” false positives.

Fall termination (`fallen_or_far_from_ladder`):

- pelvis world height `< fall_min_height_w` (default `0.25` m), or
- pelvis ladder-frame XY distance `> fall_max_ladder_distance_xy` (default `1.5` m).

Also wired:

- `time_out` (truncation);
- `nan_detection` (optional, default on);
- optional `latch_overload` when enabled in config.

## Metrics

Logged under `Episode_Metrics/`. mjlab `MetricsTermCfg.reduce` supports
`mean` and `last` only (no `max`); running maxima use a class-based term.

| Metric | Reduce | Meaning |
|--------|--------|---------|
| `max_pelvis_height_l` | last | class-based running max pelvis ladder-relative height |
| `pelvis_height_l` | last | current pelvis ladder-relative height |
| `valid_higher_attachments` | last | paid higher-attach count |
| `invalid_latch_count` | last | invalid attach attempts |
| `success` | last | binary success flag |
| `hand_contact_fraction` | mean | per-step mean hand-contact indicator |
| `time_to_success` | last | steps to first success (`-1` if none) |
| `torque_saturation_fraction` | mean | per-step torque-saturation indicator |

## Debug: `--mode rewards`

```bash
mjpython train_mimic/scripts/debug_climb.py --mode rewards --seed 42
```

Auto script phases:

1. `baseline_hold` — mostly time penalty;
2. `hand_only_no_progress` — approach without pelvis rise;
3. `attach_higher_rung` — one-off attachment bonus;
4. `wait_after_attach` — time cost accumulates, no new attach pay;
5. `raise_pelvis` — novel upward progress;
6. `invalid_latch_request` — one attempt penalty in free space;
7. `trigger_success` — physically seed top-rung connect equality + raise to goal.

Prints per-term raw values, weights, step contributions, and episode metrics.
Rewards debug env disables success/fall early terminations so the script can
finish without auto-reset loops.

## File map (Stage 5)

```text
train_mimic/tasks/climbing/
├── config/
│   ├── env.py              # wires rewards/terminations/metrics + debug env cfg
│   └── rewards.py          # ClimbingRewardConfig + configure_climbing_rewards
├── ladder/
│   └── reward_state.py     # ClimbRewardState event memory
├── mdp/
│   ├── common.py           # body/ladder height + success goal helpers
│   ├── rewards.py
│   ├── terminations.py
│   └── metrics.py
├── debug_rewards.py
└── STAGE5_REWARDS.md

train_mimic/scripts/debug_climb.py   # --mode rewards
tests/test_climbing_stage5.py
```

## Validation commands

```bash
pytest tests/test_climbing_stage0.py tests/test_climbing_stage1.py \
  tests/test_climbing_stage2.py tests/test_climbing_stage3.py \
  tests/test_climbing_stage4.py tests/test_climbing_stage5.py \
  tests/test_task_registry.py tests/test_train_script.py -q

mjpython train_mimic/scripts/debug_climb.py --mode rewards --seed 42
```

## Exit criteria (Stage 5 gate)

| Criterion | Status |
|-----------|--------|
| Independent `[B]` reward functions | Yes |
| Weights/thresholds in configuration | Yes |
| Pelvis novel-max progress (anti-farming) | Yes |
| Event/`dt` invariant magnitudes | Yes |
| Physical-time penalty | Yes |
| Attachment event memory anti-farming | Yes |
| Ladder-relative success + top-hand condition | Yes |
| Terminations: success/fall/timeout/nan | Yes |
| Metrics without zero-weight recorder | Yes |
| `debug_climb.py --mode rewards` | Yes |
| Table-driven oracle tests | Yes |
| Rewards read task state, not observations | Yes |
| `General-Tracking-G1` untouched | Yes |

## Known limitations

- Optional latch overload remains disabled by default (`break_force` still deferred).
- Foot contact fractions deferred (same as Stage 4).
- Debug success phase uses scripted pelvis raise + a physically seeded top-rung
  connect equality (not a learned policy).
- Reward debug script does not yet plot curves (console breakdown only).
- RL cfg still not trainable until Stage 8.

## Stage 6 preview

Next: static IK hold pose, `--mode hold-pose`, and brief four-contact support
validation under gravity. See `STAGE6_HOLD_POSE.md`.
