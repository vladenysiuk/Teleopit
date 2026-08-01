# Stage 3 — Attach/Detach Action and Latch Backend

Internal handoff note for `General-Climbing-G1`. Update when latch thresholds,
constraint topology, collision groups, ladder layout, or debug tooling change.

## Scope

Stage 3 delivers:

- 2-D latch action appended after 29 joint actions (`shape: 31`);
- hysteresis attach/detach commands;
- per-hand latch state (attached, rung ID, site ID, events, invalid requests);
- predeclared inactive site `connect` equalities for every hand/rung/site triple;
- `ConnectEqualityLatchBackend` driving batched `eq_active` on MuJoCo-Warp;
- `debug_climb.py --mode latch` with scripted attach/pull/detach sequence.

Stage 3 does **not** add observations, rewards, terminations, or trainable PPO
(Stage 4+ / Stage 8).

**Depends on Stage 1 ladder layout:** rungs are collision geoms on the single frame
mocap body with per-env batched `geom_pos` / `site_pos` (see
[STAGE1_LADDER.md](STAGE1_LADDER.md)). Stage 3 assumes rungs stay on the sampled
ladder after reset — no rung mocap teleport or restore on detach.

## Action contract

| Index | Term | Dim | Notes |
|-------|------|-----|-------|
| 0 | `joint_pos` | 29 | unchanged from Stage 2 |
| 1 | `latch` | 2 | left, right attach/detach commands |

Hysteresis (`LatchConfig` defaults):

| Command value | Intent |
|---------------|--------|
| `g > 0.5` | attach request |
| `g < -0.5` | detach request |
| otherwise | neutral (preserve state) |

Attach requires **contact from the previous control step** (contact sensors update
during physics). In debug/tests: touch rung with neutral latch, then send attach on
the **next** step. Same-step pose edits + attach in one step will fail attach even
if contact looks correct in the viewer.

## Constraint topology

Predeclared at scene compile via `SceneCfg.spec_fn`:

```text
latch_{hand}_rung_{id:02d}_site_{idx:02d}
  connect robot/{hand}_hand_point_site  ↔  ladder/rung_{id}_site_{idx}
```

Default counts: `2 × 12 rungs × 5 sites = 120` connect equalities (all start
inactive). Exactly **one** equality may be active per hand.

### Latch target geometry (critical)

Hand sphere radius `r_h = 0.015` m, rung radius `r_r = 0.030` m. Ordinary
contact separates geom centres by ≈ `r_h + r_r = 0.045` m. Latch sites are
**not** on the rung centreline. Each site is placed at the intended
hand-sphere centre on the robot-facing side:

```text
p_target = p_rung_axis − (r_r + r_h + δ) e_x_ladder
```

with `δ = latch_site_clearance` (default 1.5 mm). Aligning the hand-centre site
with this target leaves the sphere resting outside the cylinder so connect
equality and hand–rung collision are compatible.

Site selection on attach:

1. require hand/rung contact (`ClimbContactState`);
2. among sites within `capture_radius` (default 0.07 m), pick nearest hand site;
3. activate the matching connect constraint via `eq_active`.

Direct rung switching without detach is rejected (attached hand ignores attach
while touching another rung).

## Backend verification

MuJoCo-Warp 3.8.1 (pinned stack) supports per-world dynamic equality activation.
`ConnectEqualityLatchBackend` writes `env.sim.data.eq_active[env_ids, eq_id]`.
Connect constraints are implemented in `mujoco_warp._src.constraint` and gated by
`eq_active_in` per world.

No spring-damper fallback was introduced.

## Configuration

`train_mimic/tasks/climbing/config/latch.py`:

| Field | Default | Notes |
|-------|---------|-------|
| `attach_threshold` | 0.5 | |
| `detach_threshold` | -0.5 | must be `< attach_threshold` |
| `capture_radius` | 0.07 m | max hand→site distance; ~7 cm covers 5-site lateral gaps (~5.6 cm) plus contact |
| `break_force` | `None` | overload detach disabled until force read is reliable |
| `solref` / `solimp` | see file | connect constraint solver tuning |

Simulation capacity: `njmax = max(800, 3 × num_latch_equalities + 400)`.

## Body–ladder collision (rails + rungs)

Latch debug and training use **default ladder collision** on all active rungs
(robot body + hand points), not the hand-only probe override from
`--mode contacts`.

`config/robot.py` bit groups (MuJoCo: collide when `contype_A & conaffinity_B` in
**either** direction):

| Geom set | contype | conaffinity | Collides with |
|----------|---------|-------------|---------------|
| Robot `*_collision` | 1 | 1 \| 4 (= 5) | terrain, ladder |
| Hand points | 2 | 4 | ladder |
| Ladder rungs/rails | 4 | 1 \| 2 (= 3) | robot body, hand points |

Both ladder and robot body use `condim=3`.

**Why `ROBOT_BODY_CONAFFINITY` includes bit 4 (`LADDER_CONTYPE`):** CPU MuJoCo
accepts one-sided filter matches; MuJoCo-Warp broadphase needs **both** directions
non-zero for reliable body↔ladder contacts. Hand points already had symmetric
masks (`contype=2`, `conaffinity=4` and reverse).

**Rails vs rungs (same frame body):**

| Piece | Position source | Viewer sync needed? |
|-------|-----------------|---------------------|
| Rails | Fixed `geom_pos` in MJCF | No |
| Rungs | Runtime batched `geom_pos` | Yes — see below |

Rails and rungs are both parented on the frame mocap body. After the Stage 1
restructure (rungs on frame, not per-rung mocap bodies), rung hand contacts and
body contacts both work in Warp when groups and `condim` are set as above.

## Ladder runtime + native viewer

`LadderRuntime.attach()` registers batched layout fields:

```python
env.sim.expand_model_fields(("geom_pos", "site_pos"))
```

Without this, the native MuJoCo viewer keeps rungs at compiled default local
`(0, 0, 0)` — stacked on the frame origin — while Warp physics uses the sampled
heights from `LadderRuntime.apply()`. Symptom: **rails visible, rungs invisible**,
robot appears to walk through empty space where rungs should be.

mjlab viewers copy registered fields from `sim.model[env_idx]` into the host
`MjModel` each frame (`sim.expanded_fields & VIEWER_MODEL_FIELDS`). Rails do not
need runtime updates; rungs do.

## Debug: `--mode latch`

```bash
mjpython train_mimic/scripts/debug_climb.py --mode latch --seed 42
mjpython train_mimic/scripts/debug_climb.py --mode latch --seed 42 --manual-latch
```

### Auto script phases

Rungs never move after env reset. The **robot** moves toward fixed ladder rungs
for approach/contact; detach does **not** restore or recreate rungs.

| Phase | Motion | Latch L/R | Notes |
|-------|--------|-----------|-------|
| `settle_near_rung_a` | slide | 0 / 0 | after init seed near rung 2 |
| `left_approach_rung_a` | slide → rung 2 | 0 / 0 | |
| `left_touch` / `left_attach` | slide (slow) | 0→1 / 0 | attach after contact |
| `left_pull_demo_a` | pull **away** (−ladder X) | 1 / 0 | hand stays latched |
| `left_detach` | hold | −1 / 0 | release |
| `left_approach_rung_b` / `left_reattach` | slide → rung 3 | 0→1 / 0 | re-seed + attach while contact live |
| `left_pull_demo_b` | pull away | 1 / 0 | second latched pull |

Init teleports the root so the left hand starts ~12 cm in front of rung 2
(avoids a long fall-prone walk from the default spawn). The auto script is
**left-hand only**; use `--manual-latch` for the right latch.

Default debug ladder: 6 active rungs; script uses rung **2** then **3**.

Manual keys (with `--manual-latch`): `A` attach left, `D` detach left, `=` neutral.

### Debug root motion (`debug_probe.py`)

Implemented in `nudge_robot_toward_rung` / `nudge_robot_away_from_ladder`.

**Do not teleport the root during latch debug.** An earlier implementation used
`write_root_link_pose_to_sim` every step plus zero velocity. That **ignores
collision entirely**, so the robot visually passed through rails and rungs even
when Warp contact filters were correct.

Current behavior:

1. **Velocity drive** — set root linear velocity for one physics substep
   (`step_m / sim.timestep`); let contacts resolve during `env.step()`.
2. **Bay-aware approach** — decompose motion in the ladder frame:
   - always apply the **forward** (+ladder X) component toward the rung;
   - apply **lateral / vertical** correction only once
     `root_x ≥ frame_x − 0.12` (pelvis inside the ladder bay).
   - avoids diagonal paths that cut through the **side rails** from outside.
3. **Rail velocity clamp** — zero lateral root velocity that would drive the body
   through a side rail from outside (`±rail_half_width ± rail_radius − margin`).

`--mode contacts` still moves a **probe rung** toward the hand (hand-only
collision on that rung). `--mode latch` does **not** move rungs.

### Ladder geometry mental model (debug seed 42)

Typical debug layout (`ladder_distance_range=(0.85, 0.85)`):

- Frame mocap ≈ world `x=0.85`;
- rails at `y=±0.24`, vertical cylinders;
- rung centers span `y≈0` (horizontal cylinders);
- robot starts near origin, `y≈0`.

Approach from the front goes **between** the rails (`|y| < 0.24`). Approaching
from the side must not cut through a rail cylinder; bay-aware + velocity motion
is meant to enforce that. Walking through the **gap** between rails (not hitting
either rail) is expected.

## Troubleshooting and confusing points

Collected from Stage 3 latch/collision debugging. Read before changing ladder,
collision, viewer, or debug motion code.

### “Rungs invisible but rails show”

- **Cause:** viewer not syncing batched `geom_pos` (missing
  `expand_model_fields`).
- **Fix:** `LadderRuntime.attach()` registration; verify
  `"geom_pos" in env.sim.expanded_fields` after reset.
- **Not:** missing rung geoms; Warp `geom_xpos` was already correct.

### “Robot passes through rails / rungs” in `--mode latch`”

- **Often cause #1:** kinematic root teleport (historical bug) — fixed by velocity
  drive. Re-check any new debug helper does not call `write_root_link_pose_to_sim`
  every step for approach.
- **Often cause #2:** diagonal approach from outside the bay through a side rail
  while teleporting or forcing high lateral velocity every step.
- **Not necessarily:** missing collision groups — verify with embedded pose +
  physics-only steps (no per-step pose override). Warp `nacon` and contact sensors
  can show `torso_collision ↔ rail_left` penetration.

### `mj_data.ncon` vs Warp contacts

- `env.sim.mj_data` is a **CPU template**; it is **not** kept in sync with Warp
  contact manifolds each step.
- Use **`sim.data.nacon`** and `sim.data.contact` for Warp-side pairs, or
  **contact sensors** / `ClimbContactState` for hand↔rung identity.
- CPU `mj_collision` on a synced snapshot (qpos/mocap + expanded `geom_pos`) is
  useful for offline verification, not as the in-loop debug readout.

### Attach fails though hands “touch” in the viewer

1. **One-step delay:** contact is from the **previous** step; send attach one
   step after contact appears in `ClimbContactState`.
2. **Script too short:** velocity approach is slower than old teleport; increase
   `left_approach_*` / `left_retouch` steps (see table above).
3. **Wrong rung / inactive rung:** inactive rungs parked at `inactive_rung_pose`
   (`z=-50`); contacts on them are ignored.

### “Pull” vs “detach” naming

| User phrase | Script phase | What moves |
|-------------|--------------|------------|
| Detach / release | `left_detach` (`latch_left=-1`, hold) | constraint off; rung stays |
| Pull away (latched) | `left_pull_demo` (`pull_robot`) | robot root velocity away |
| Approach | `slide_robot_to_rung` | robot toward fixed rung |

Detach does **not** move rungs off the ladder or restore mocap poses.

### Per-rung mocap bodies (historical)

Do **not** revert to per-rung mocap bodies for “easier” debug motion. Secondary
mocap rung bodies rendered but **did not generate reliable contacts** on the
mocap body in Warp; the frame-parented `geom_pos` layout is the supported
topology (Stage 1).

### Entity init vs frame mocap

Ladder entity `init_state.pos` uses `ladder_distance` (default 0.80); sampled
`frame_pos.x` uses `ladder_distance_range` (debug: 0.85). World pose is driven
by **mocap** each reset; entity body pos is not the whole story. Do not “fix”
visual/collision mismatch by doubling distance in both places without checking
`geom_xpos`.

### Tuning script timing

If reattach or first attach consistently misses contact:

- increase **`left_detach`** hold (settle after release);
- increase **`left_approach_rung_b`** / **`left_retouch`** (or symmetric first
  approach/touch phases);
- optionally increase `approach_step_m` slightly — trades speed vs rail impact.

Values in the phase table above are tuned for **velocity** approach at
`sim.timestep=0.005`, `decimation=1`.

## File map (Stage 3)

```text
train_mimic/tasks/climbing/
├── config/
│   ├── env.py            # latch action, spec_fn, njmax, latch env cfg
│   ├── latch.py          # LatchConfig
│   └── robot.py          # collision bit groups, condim, ladder cfg
├── ladder/
│   ├── latch.py          # topology, backend, ClimbLatchState
│   └── state.py          # LadderRuntime, expand_model_fields for rung layout
├── mdp/
│   ├── actions.py        # LatchActionCfg / LatchAction
│   └── events.py         # attach/reset latch
├── debug_probe.py        # nudge_robot_*, bay-aware velocity approach
├── debug_latch.py        # scripted latch debug helpers
└── STAGE3_LATCH.md

train_mimic/scripts/debug_climb.py   # --mode latch
tests/test_climbing_stage3.py
tests/test_climbing_stage2.py        # body-rail repulsion, hand contact
```

## Reset ordering

Safe order (wired in `make_general_climbing_env_cfg` event dict):

1. deactivate equality constraints (`reset_climb_latch`);
2. clear latch + contact Python state;
3. resample / move ladder sites (`reset_ladder`);
4. mjlab `sim.reset` / `scene.reset` already ran just before events (robot state);
5. `sim.forward()` + `sense()` refresh derived contact state after `_reset_idx`.

`ClimbLatchState.reset` deactivates `eq_active` **before** clearing Python IDs.

## Geometry diagnostics

`ClimbLatchState.geometry_diagnostics()` logs during attach/pull:

| Metric | Meaning |
|--------|---------|
| `hand_to_rung_axis_dist` | hand-sphere centre → rung axis |
| `signed_separation` | axis distance − `(r_r + r_h)`; negative = penetration |
| `contact_normal_force` | contact-frame `|force_x|` |
| `equality_residual` | `‖hand_site − attached_rung_site‖` |

Gate targets: steady penetration ≲ a few mm, bounded contact force, equality
residual consistent with the external target site, no NaNs/jitter.

## Batched per-world isolation

Automated coverage in `test_per_world_latch_isolation` (4 envs):

| Env | Command |
|-----|---------|
| 0 | left hand → rung 2 |
| 1 | left hand → rung 5 |
| 2 | right hand → rung 3 |
| 3 | no attachment |

Verifies intended `eq_active[env, eq_id]` only, subset reset of envs 1–2, and
independent per-env `site_pos` / geom layout.

## Validation commands

```bash
pytest tests/test_climbing_stage0.py tests/test_climbing_stage1.py \
  tests/test_climbing_stage2.py tests/test_climbing_stage3.py \
  tests/test_task_registry.py tests/test_train_script.py -q

mjpython train_mimic/scripts/debug_climb.py --mode latch --seed 42
```

Useful one-off checks while tuning collision or script timing:

```bash
# Hand contact on fixed ladder (Stage 2 path)
mjpython train_mimic/scripts/debug_climb.py --mode contacts --seed 42

# Ladder layout + viewer sync (Stage 1 path)
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42
```

## Exit criteria (Stage 3 gate)

| Criterion | Status |
|-----------|--------|
| 2-D latch action after joint actions | Yes (31-D total) |
| Hysteresis attach/detach/neutral | Yes |
| Per-hand latch state + events | Yes |
| No direct rung-i → rung-j switch | Yes |
| Predeclared site connect constraints | Yes (120 default) |
| Closest eligible site selection | Yes |
| External latch targets (not centreline) | Yes (`latch_site_normal_offset`) |
| Geometry diagnostics on attach/pull | Yes |
| Batched per-world `eq_active` isolation | Yes (4-env test) |
| Safe reset order (deactivate → clear → resample) | Yes |
| `ConnectEqualityLatchBackend` on MuJoCo-Warp | Yes (`eq_active`) |
| Transition table tests | Yes |
| Reset clears constraints + Python state | Yes |
| `debug_climb.py --mode latch` | Yes |
| Body + hand ladder collision (not hand-only) | Yes |
| Native viewer shows runtime rung positions | Yes (`expand_model_fields`) |
| `General-Tracking-G1` untouched | Yes |

**Owner gate:** Stage 3 remains open until owner review of pull-demo geometry
metrics (penetration / residual / force) and the batched isolation evidence.

## Known limitations

- Attach uses previous-step contact (one-step delay vs. same-step pose edits).
- `break_force` overload detach not implemented (explicitly deferred).
- Debug script drives the robot with **root velocity**, not joint teleop — that
  is a diagnostic tool, not evidence that the actuated G1 can perform the motion.
- `--mode contacts` still moves probe rung geoms; `--mode latch` does not.
- Forced root velocity every step can still slowly penetrate rails if lateral
  clamp and bay logic are bypassed; training policy motion is unaffected.
- 120 precompiled equalities × 3 rows each — monitor `njmax` if topology grows.
- Placeholder RL cfg still not trainable.
- Capture radius may be tightened further from measured hand→site distances.

## Stage 4 preview

Next: observation groups, relative-rung provider, ladder encoders, and
`debug_climb.py --mode observations`. Keep rewards/terminations independent of
actor observations.
