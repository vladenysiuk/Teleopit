---
sidebar_position: 7
---

# Climbing Simulator Reference

Implementation reference for the **simulator-side** of `General-Climbing-G1`: ladder generation, G1 integration, contacts, latches, observations (geometry providers), hold pose, and reset order.

Reward/termination arithmetic and PPO wiring are documented in [Climbing RL Reference](./climbing-rl.md). Usage commands are in the [Climbing Tutorial](../tutorials/climbing.md).

## Architecture overview

```text
LadderConfig + LadderGenerator
        │
        ▼
  build_ladder_spec()  ──► fixed MJCF topology (one mocap frame, max_rungs cylinders)
        │
        ▼
  LadderRuntime (per env) ──► batched geom_pos / site_pos + frame mocap pose at reset
        │
        ├──► ClimbContactState   (hand ↔ rung identity from contact sensors)
        ├──► ClimbLatchState     (connect equality activation per hand)
        └──► RelativeRungProvider (privileged torso-frame rung endpoints for obs)

G1 (canonical g1_29dof.xml) + hand point spheres + PD joint actuators
        │
        ▼
  MuJoCo / mujoco-warp @ 200 Hz  ──►  policy @ 50 Hz (decimation=4)
```

**Fixed design decisions (simulator):**

1. Rungs use **pure cylindrical collision** geometry with high friction. No invisible flat support surfaces.
2. Ladder topology is **compiled once** (`max_rungs`). Active rung count, heights, and frame pose are sampled at reset via batched writes — no per-episode MJCF recompile.
3. **One mocap frame body** carries all rung geoms and attachment sites (not per-rung mocap bodies; Warp contact reliability).
4. Latch targets sit at **external hand-sphere centres**, offset from the rung axis toward the robot:
   `p_target = p_rung_axis − (r_rung + r_hand + δ) e_x_ladder`.

## Package map (simulator)

| Path | Role |
|------|------|
| `config/ladder.py` | `LadderConfig` — rung count/radius/spacing, friction, pose ranges |
| `config/robot.py` | G1 climbing spec: hand point geoms, collision bit groups |
| `config/scene.py` | Scene-only env for Stage 1 ladder viewer |
| `ladder/geometry.py` | `CylinderRungGeometry` factory (only supported kind) |
| `ladder/generator.py` | `build_ladder_spec()` — fixed MJCF ladder |
| `ladder/state.py` | `LadderRuntime`, `LadderSampler` — per-reset sampling |
| `ladder/contacts.py` | `ClimbContactState` — batched contact identity |
| `ladder/latch.py` | `ClimbLatchState`, `add_latch_equalities_to_spec()` |
| `ladder/relative_rungs.py` | `RelativeRungProvider` — K-rung window in torso frame |
| `ladder/hold_ik.py` | Static four-contact IK for hold-pose / curriculum reset |
| `mdp/actions.py` | `LatchAction` — 2D attach/detach with hysteresis |
| `mdp/observations.py` | Observation term functions (proprio + relative rungs) |
| `mdp/events.py` | Reset/startup event hooks for sim adapters |
| `config/observations.py` | Observation group wiring, `LadderObservationConfig` |
| `debug_*.py` | Stage-specific debug helpers and visualizers |

## G1 robot integration

**Asset:** `assets/robots/unitree_g1/g1_29dof.xml` (same canonical XML as tracking).

**Hand contact points** (`config/robot.py`):

| Body | Geom | Offset (body frame) | Radius |
|------|------|---------------------|--------|
| `left_wrist_yaw_link` | `left_hand_point` | `(0.18, -0.025, 0.0)` | 15 mm |
| `right_wrist_yaw_link` | `right_hand_point` | `(0.18, 0.025, 0.0)` | 15 mm |

Colocated attachment **sites** (`left_hand_point_site`, `right_hand_point_site`) are latch equality anchors.

**Collision filtering** uses MuJoCo bit groups:

- Robot body geoms collide with ladder rungs and each other (standard).
- Hand point geoms (`contype=2`) collide with ladder (`conaffinity` includes ladder bit).
- Hand points are isolated from most robot self-collision to avoid false positives.

## Ladder specification and sampling

### `LadderConfig` defaults (production)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_rungs` | 12 | Compiled topology cap |
| Active rung count | 4–10 | Sampled per reset |
| `rung_radius` | 30 mm | `geometry_kind="cylinder"` only |
| `rung_length` | 450 mm | |
| Spacing | 220–280 mm | Strictly increasing active heights |
| `sites_per_rung` | 5 | Latch targets along rung Y |
| `inactive_rung_pose` | z = −50 m | Parks unused rungs outside scene |

### Fixed topology (`ladder/generator.py`)

- Single **mocap frame** body with all rung cylinder geoms + rail visuals.
- Stable IDs: `rung_00` … `rung_{max-1}` regardless of sampled height.
- Attachment sites offset toward robot by `rung_radius + hand_radius + latch_site_clearance`.

### Per-reset sampling (`ladder/state.py`)

`LadderSampler.sample(num_envs)` returns:

- `active_mask[B, max_rungs]`
- `rung_heights[B, max_rungs]` (ladder-relative)
- Frame position, yaw, pitch
- Writes batched `geom_pos`, `site_pos`, and mocap pose — **no recompile**.

Inactive rungs are moved to `inactive_rung_pose` and masked out of contact/observation logic.

## Contact sensing

**Sensors:** one `ContactSensorCfg` per `(hand, rung_id)` pair — up to `2 × max_rungs` sensors per env.

**Adapter:** `ClimbContactState` (`ladder/contacts.py`) exposes batched:

| Field | Shape | Notes |
|-------|-------|-------|
| `hand_in_contact` | `[B, 2]` | Left/right boolean |
| `hand_rung_id` | `[B, 2]` | `-1` if none |
| `hand_rung_height` | `[B, 2]` | Sampled ladder-relative height |
| `hand_contact_force` | `[B, 2]` | Normal force (debug) |
| `hand_contact_pos` | `[B, 2, 3]` | World position (debug) |

**Multiple-contact rule:** prefer matching rung with **maximum normal force**; documented tie behaviour in Stage 2 tests.

Inactive rungs never register contact.

## Latch mechanism

### Action (`mdp/actions.py`)

Two-dimensional latch command appended after 29 joint position actions:

| Command range | Threshold | Effect |
|---------------|-----------|--------|
| `g > 0.5` | attach | Request attach (needs valid contact + capture radius) |
| `g < -0.5` | detach | Request detach |
| between | neutral | Preserve current state |

Direct rung-to-rung switching is **disallowed**; detach first.

### Backend (`ladder/latch.py`)

- Predeclared **`connect` equalities** for every valid `(hand, rung, site)` triple; exactly one activated per attached hand.
- Nearest eligible site selected on attach.
- `LatchConfig.capture_radius` default **0.07 m**.
- `LatchConfig.break_force` default **500 N**: each physics substep reads the active connect equality force from `efc.force` and detaches that hand when `||F||` exceeds the threshold. `None` disables overload break (debug / hold-pose only).
- Default `solref=(0.05, 1.0)` keeps the latch soft enough to limit attach impulse and stored elastic energy. An unbreakable stiff connect is an infinite-strength weld to the kinematic ladder frame and can slingshot the free base.

### Per-env state

| Field | Description |
|-------|-------------|
| `attached[B, 2]` | Boolean latch state |
| `rung_id[B, 2]` | Attached rung or `-1` |
| `site_id[B, 2]` | Selected site index |
| `capture_radius[B]` | Per-env radius (curriculum-ready) |
| `equality_force[B, 2]` | Latest connect equality force magnitude (N) |
| `overload_detach_event[B, 2]` | True on the substep a hand broke from overload |

## Observations (simulator providers)

Observations are grouped in `config/observations.py`. **Rewards read task state, not observation tensors.**

### Actor proprioception (101D flat + 10-frame history)

| Term | Dim |
|------|----:|
| `robot_joint_pos_rel` | 29 |
| `robot_joint_vel` | 29 |
| `robot_base_ang_vel_b` | 3 |
| `robot_projected_gravity_b` | 3 |
| `prev_joint_action` | 29 |
| `prev_latch_action` | 2 |
| `hand_in_contact` | 2 |
| `hand_attached` | 2 |
| `attached_rung_rel_height` | 2 |

Detached hands: `attached_rung_rel_height = 0` with `attached = 0`.

### Actor ladder — relative rungs (`ladder/relative_rungs.py`)

- **K = 6** slots: 2 below + 4 above pelvis **ladder-relative height**
  (`h = u_L^T (p − p_L)`, not world Z).
- Per slot: `endpoint_a_torso(3) + endpoint_b_torso(3) + valid(1)` = **7 features**.
- Transform: `p_torso = R_world_torso^T (p_world − p_torso_world)`.
- Inactive/ out-of-window rungs: `valid=0`, finite padding.

### Critic-only privileged state

Base linear velocity, pelvis height, ladder frame pose in torso frame, active rung heights with explicit validity mask.

### Ladder perception modes

| Mode | Status |
|------|--------|
| `relative_rungs` | Production — privileged geometry |
| `depth` | **Not implemented** — fails at env construction until `DepthCameraProvider` exists |

Depth encoder interface exists in `rl/ladder_encoders.py` for future wiring only.

## Hold pose and static validation

**IK:** `ladder/hold_ik.py` — least-squares pose targeting feet on rung tops, hands at higher rung latch sites, upright torso, joint-limit regularization.

**Debug:** `--mode hold-pose` simulates 2.5 s gravity hold with both hands force-attached.

**Artifact:** `data/hold_pose_seed42.npz` — validated debug seed only.

Stage 6 acceptance thresholds (sanity checks, not research claims):

- Hold ≥ 2 s, pelvis drop &lt; 10 cm, foot slip &lt; 5 cm, both feet retain contact.

## Reset order and sim capacities

Reset events in `config/env.py` (order matters):

1. `reset_climb_latch` — deactivate all connect equalities
2. `reset_climb_contacts` — clear contact adapter
3. `reset_ladder` — resample ladder geometry
4. `reset_climb_rewards` — clear event memory (see RL doc)
5. Curriculum physics / pose (easy preset: hold IK + latch seed)

**Capacity guards** (`config/env.py`):

| Buffer | Typical value | Notes |
|--------|---------------|-------|
| `nconmax` | 384–768 per env | Higher for hold-pose easy preset |
| `njmax` | 1200–2560 per env | Predeclared latch rows + contacts |

Stage 7 validates batches up to **64 envs/GPU** on CUDA with positive contact/constraint headroom.

## Debug visualizers

| Module | Mode |
|--------|------|
| `debug_vis.py` | Ladder sites, contact spheres, latch highlights |
| `debug_observations.py` | Torso-to-rung endpoint debug lines |
| `debug_probe.py` | Movable probe rung for contact mode |

## Extension points

| Change | Where to start |
|--------|----------------|
| New rung geometry profile | `ladder/geometry.py` factory + `LadderConfig.geometry_kind` |
| Depth camera perception | Implement `DepthCameraProvider`, wire `ladder_observation.mode="depth"` |
| Foot attachment | New equality backend; not in v1 scope |
| Per-episode MJCF | **Not supported** by design — use batched state instead |

## Related stage gate notes

Detailed validation evidence per stage:

| Stage | Doc |
|------:|-----|
| 0 | `STAGE0_BASELINE.md` |
| 1 | `STAGE1_LADDER.md` |
| 2 | `STAGE2_CONTACTS.md` |
| 3 | `STAGE3_LATCH.md` |
| 4 | `STAGE4_OBSERVATIONS.md` |
| 6 | `STAGE6_HOLD_POSE.md` |
| 7 | `STAGE7_RESET_MDP.md` |

Handoff summary: `STAGE10_HANDOFF.md`.
