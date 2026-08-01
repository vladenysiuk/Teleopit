# Stage 2 — Climbing Task Skeleton, G1 Integration, and Point-Hand Contacts

Internal handoff note for `General-Climbing-G1`. Update this file when contact
groups, sensor wiring, probe debug behavior, or registration change.

## Scope

Stage 2 delivers:

- `General-Climbing-G1` registration (skeleton env + placeholder RL cfg);
- canonical 29-DoF G1 with joint-position actions and point-hand collision geoms;
- collision filtering between robot body, hand points, and ladder rungs;
- per-hand / per-rung contact sensors and `ClimbContactState` identity adapter;
- `debug_climb.py --mode contacts` with a kinematic mocap probe rung (no IK).

Stage 2 does **not** implement latch actions, observations, rewards,
terminations, or trainable PPO (Stage 3+ / Stage 8).

## Fixed contracts (do not break)

| Item | Value |
|------|-------|
| Robot | Canonical `assets/robots/unitree_g1/g1_29dof.xml` via `make_climbing_robot_cfg()` |
| Control | `JointPositionActionCfg`, 29-D actions, `G1_ACTION_SCALE` |
| Wrist bodies | `left_wrist_yaw_link`, `right_wrist_yaw_link` |
| Hand point offsets (body frame) | left `(0.18, -0.025, 0)`, right `(0.18, 0.025, 0)` |
| Hand geoms | `left_hand_point`, `right_hand_point` spheres, radius **0.015 m** |
| Hand geom margin/gap | `0` (no early-contact margin) |
| Ladder topology | **One** frame mocap body; 12 rung geoms + sites via batched `geom_pos` / `site_pos` (see Stage 1 revision note) |
| Heights / active mask | From `LadderRuntime.sample` only — **never** reconstruct from geom poses |
| Ladder geometry | Same fixed topology as Stage 1; probe debug may temporarily move one rung geom |

## Collision groups

Bitmasks in `config/robot.py` (MuJoCo: collide when `contype_A & conaffinity_B`):

| Geom set | contype | conaffinity | Collides with |
|----------|---------|-------------|-----------------|
| Robot `*_collision` | 1 | 5 (`1\|4`) | Terrain, ladder (symmetric Warp filter) |
| Hand points | 2 | 4 | Ladder only (not robot self) |
| Ladder (entity `CollisionCfg`) | 4 | 3 (`1\|2`) | Robot body + hand points |

Feet/body can still touch rungs. Hand points do not self-collide with the robot.
`ROBOT_BODY_CONAFFINITY` includes the ladder bit so mujoco_warp broadphase sees
both filter directions; permanent regression tests cover body–rung, body–rail,
hand–rung, hand–robot noncollision, and inactive-rung noncollision.

## Contact sensing

- One `ContactSensorCfg` per `(hand, rung_id)` → **24** sensors (`12 × 2`).
- Fields: `found`, `force`, `pos`; `reduce="maxforce"`, `num_slots=1`.
- `ClimbContactState` (in `env.extras["climb_contact_state"]`) exposes batched:

  | Tensor | Shape | Notes |
  |--------|-------|-------|
  | `hand_in_contact` | `[B, 2]` bool | left=0, right=1 |
  | `hand_rung_id` | `[B, 2]` int64 | `-1` = none |
  | `hand_rung_height` | `[B, 2]` float | from `LadderRuntime.sample.rung_heights` |
  | `hand_contact_force` | `[B, 2]` float | contact-frame normal `\|force[..., 0]\|` |
  | `hand_contact_pos` | `[B, 2, 3]` float | debug |

- Contact sensors are configured with **`global_frame=False`** so `force` is in
  the MuJoCo contact frame (column 0 = normal, primary → secondary). Do not
  treat `|force_x|` as a world-axis quantity.
- **Multi-contact rule:** among **active** rungs with `found > 0`, pick max
  contact-frame normal force `|force[..., 0]|`; ties prefer the **lower** rung
  index.
- Inactive rungs are masked out even if a sensor fires.
- Reset clears cached contact state via `reset_climb_contact_state`.

## Debug: `--mode contacts`

### What it is

> **Topology note (Stage 3 revision):** there is no separate “mocap rung body.”
> The probe teleports one **rung collision geom** (and its latch sites) by
> writing batched `model.geom_pos` / `model.site_pos` on the shared frame mocap
> body. Visually it leaves the rails; that is intentional for contact tests.

A selected rung geom is moved each step near one hand for contact identity
checks. It is **not** kept on the ladder rails (by design). There is no latch.
Robot dynamics stay **live** (zero joint actions → robot falls under gravity;
useful for stress-testing contact identity).

### Probe placement rules (`debug_probe.py`)

1. Approach along **ladder-frame +X** (in front of the hand), not into the torso.
2. **Lateral offset** along rung axis (`0.35 × rung_length`) so the 0.45 m
   cylinder covering one hand does not also hit the other (~0.25 m apart).
3. Probe geom `conaffinity` set to hand-point-only for that rung so the cylinder
   does not shove through legs/torso while following a falling robot.
4. Debug highlight uses the **live geom centre**, not `sample.rung_pos_w` (the
   parked ladder slot), so the green/red sphere follows the probe.

Default: probe hand = **left**, `--probe-rung 3`, `--probe-separation 0.08`.
Contact typically appears near **~0.04–0.05 m** separation for default radii.

### Commands

```bash
# macOS: native viewer requires mjpython
mjpython train_mimic/scripts/debug_climb.py --mode contacts --seed 42

mjpython train_mimic/scripts/debug_climb.py --mode contacts --seed 42 \
  --probe-rung 3 --probe-separation 0.08

# Quieter GPU / fewer overlay spheres
mjpython train_mimic/scripts/debug_climb.py --mode contacts --seed 42 --no-debug-vis
```

### Viewer controls (contacts)

| Key | Climbing / mjlab action | MuJoCo built-in (also fires) |
|-----|-------------------------|------------------------------|
| `↑` / `↓` | Probe separation ±5 mm | — |
| `=` / `-` | Probe separation ±1 mm | Speed up / slow down |
| `,` / `.` | Probe hand left / right | Prev / next env (noop if `num-envs=1`) |
| `Enter` | Env reset | — |
| `Space` | Pause / resume | — |
| `→` | Single-step (when paused) | — |
| `R` | Toggle mjlab debug overlay spheres | *(not a MuJoCo vis flag)* |
| `P` | Toggle reward plots | **Contact Split** vis — plots empty (no rewards yet) |
| `N` | *(unused in contacts)* | **Island** vis (often yellow body tint) |
| `I` | — | **Inertia** boxes (yellow) |
| `F1` | — | MuJoCo shortcut help |
| `Q` / close | Quit | — |

If `N` turns the robot yellow: that is MuJoCo Island viz, not a climbing bug —
press `N` again to clear. Prefer `--verbose` to see mjlab `[INFO]` toggles.

### Manual checklist

1. Start at separation ~0.08 m → `contact=false`, `rung_id=-1`.
2. `↓` until contact → correct hand, rung ID, height from `LadderRuntime`.
3. Move away → contact clears promptly.
4. `.` then probe right hand; `,` for left. Only the **selected** hand should
   report contact for the probe rung.
5. Pause / single-step near the boundary; watch for flicker or stale IDs.
6. Trust console contact lines and yellow contact-point spheres over distant
   ladder geometry (probe is intentionally off-ladder).

## Registration and RL placeholder

- Registered via `train_mimic/tasks/climbing/config/registry.py` (imported from
  `train_mimic.tasks`).
- `config/rl.py` is a **stub** so `register_mjlab_task` has an `rl_cfg`. It is
  not trainable: empty `obs_groups`, `max_iterations=1`, `runner_cls=None`.
- `train_mimic.app.SUPPORTED_TASKS` is still tracking-only; do not train climbing
  through `train.py` until Stage 8 wires a real model and adds the task ID.

## File map (Stage 2)

```text
train_mimic/tasks/climbing/
├── config/
│   ├── env.py            # make_general_climbing_env_cfg, make_climbing_contacts_env_cfg
│   ├── robot.py          # hand points + collision groups
│   ├── rl.py             # placeholder PPO cfg (registry only)
│   └── registry.py       # register_mjlab_task
├── ladder/
│   └── contacts.py       # ClimbContactState
├── mdp/events.py         # reset_ladder_sample, attach/reset climb contacts
├── debug_probe.py        # probe placement, hand-only collision helper
├── debug_vis.py          # scene + contact overlays
└── STAGE2_CONTACTS.md

train_mimic/scripts/debug_climb.py   # --mode scene | contacts
tests/test_climbing_stage2.py
```

## Validation

```bash
pytest tests/test_climbing_stage0.py tests/test_climbing_stage1.py \
  tests/test_climbing_stage2.py tests/test_task_registry.py tests/test_train_script.py -q
```

## Exit criteria (Stage 2 gate)

| Criterion | Status |
|-----------|--------|
| `General-Climbing-G1` registered | Yes |
| Point-hand geoms on canonical G1 | Yes |
| Hand/rung contact identity (ID + height) | Yes |
| Inactive rungs never reported | Yes |
| Deterministic multi-contact reduction | Yes (docs + unit test) |
| `debug_climb.py --mode contacts` | Yes |
| `General-Tracking-G1` untouched | Yes |
| No ladder geometry / latch changes | Yes |

## Known limitations

- No latch, observations, rewards, or terminations.
- Placeholder RL cfg — do not run PPO yet.
- Probe is one hand / one rung at a time; not a climbing policy demo.
- Live robot + zero actions → fall is expected in contacts mode.
- Clearance ~0.05 m along ladder +X (geometry-dependent).
- 24 contact sensors per env (may optimize later).
- Several letter keys conflict with MuJoCo visualize flags (`N`, `P`, `I`, …).

## Stage 3 preview

Next: attach/detach latch action, hysteresis, constraint backend check, and
`debug_climb.py --mode latch`. Keep contact identity and `LadderRuntime` as the
source of rung height / active mask. Do not alter ladder geometry without owner
approval.
