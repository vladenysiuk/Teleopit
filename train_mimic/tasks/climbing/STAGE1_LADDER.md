# Stage 1 — Ladder Specification, Generation, and Manual Viewer

Internal implementation note for `General-Climbing-G1`. Recorded after Stage 1
completion; update when ladder topology, sampling, or debug tooling change.

## Scope

Stage 1 delivers:

- a configurable fixed-topology ladder (`LadderConfig`);
- pure cylindrical rung collision geometry via a replaceable factory;
- procedural per-environment sampling (heights, active mask, frame pose);
- batched mocap updates without MJCF recompilation on reset;
- a scene-only mjlab env for inspection (`make_ladder_scene_env_cfg`);
- `debug_climb.py --mode scene` with native viewer and optional MJCF export.

Stage 1 does **not** register `General-Climbing-G1`, add the G1 robot, or
implement contact/latch/MDP terms (Stage 2+).

## Specifications

### Fixed topology (`LadderConfig` defaults)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_rungs` | 12 | Compiled body/geom/site count; stable IDs `rung_00` … `rung_11` |
| `min_active_rungs` | 4 | Lower bound on sampled active count |
| `max_active_rungs` | 10 | Upper bound on sampled active count |
| `rung_radius` | 0.030 m | Cylinder radius (~30 mm) |
| `rung_length` | 0.45 m | Rung span along local Y after axis rotation |
| `rung_friction` | `(1.2, 0.005, 0.0001)` | Sliding / torsion / rolling (high sliding friction) |
| `base_height` | 0.30 m | Height of first active rung in ladder frame |
| `spacing_min` | 0.22 m | Min vertical gap between consecutive active rungs |
| `spacing_max` | 0.28 m | Max vertical gap between consecutive active rungs |
| `rail_radius` | 0.020 m | Vertical rail cylinder radius |
| `rail_half_width` | 0.24 m | Rail centerline Y offset (± from ladder frame) |
| `sites_per_rung` | 5 | Attachment sites evenly spaced along rung length |
| `latch_site_clearance` | 0.0015 m | Extra gap beyond `r_rung + r_hand` for latch targets |
| `inactive_rung_pose` | `(0, 0, −50)` | Local parking pose for inactive rungs |
| `geometry_kind` | `"cylinder"` | Only `"cylinder"` supported; other kinds fail fast |

Rails span the **full compiled height**:

```text
max_ladder_height = base_height + (max_rungs − 1) × spacing_max
                  ≈ 0.30 + 11 × 0.28 = 3.38 m   (defaults)
```

Active rungs are placed bottom-up from `base_height` with random gaps; they do
**not** automatically fill the full rail height unless the sampled active count
and spacings happen to reach it.

### Randomized sampling ranges (per env, every reset)

All draws are uniform unless noted. Angles are radians.

| Quantity | Config field | Range |
|----------|--------------|-------|
| Active rung count | `min_active_rungs`, `max_active_rungs` | 4 – 10 (integer) |
| Inter-rung spacing | `spacing_min`, `spacing_max` | 0.22 – 0.28 m |
| Ladder distance (world X) | `ladder_distance_range` | 0.60 – 1.00 m |
| Ladder yaw (about world Z) | `ladder_yaw_range` | −0.30 – +0.30 rad (~±17°) |
| Ladder pitch / tilt (about Y) | `ladder_tilt_range` | −0.05 – +0.05 rad (~±2.9°) |

Fixed on reset (not randomized): first rung height (`base_height`), rung
radius/length/friction, rail geometry, site count.

`ladder_distance = 0.80` in `LadderConfig` is only the entity `init_state` default;
after reset, distance comes from `ladder_distance_range`.

### Rung geometry contract (plan fixed decision #1)

- Rungs use **pure cylindrical collision** geoms (no invisible flat support).
- Geometry is selected through `create_rung_geometry(kind)` so a different
  profile can be substituted later.
- MuJoCo default cylinder axis is +Z; rungs are rotated +90° about X so the
  cylinder spans local **+Y** (ladder width direction).

### Attachment sites

- Names: `rung_{id:02d}_site_{idx:02d}` (stable across samples).
- Lateral positions: evenly from `−rung_length/2` to `+rung_length/2` along local Y.
- Normal offset: sites sit at ladder-frame
  `x = −(rung_radius + hand_point_radius + latch_site_clearance)` relative to
  the rung axis (robot-facing side). Default clearance is 1.5 mm. This is the
  intended hand-sphere centre for Stage 3 connect latches so equality and
  collision do not fight at the centreline.
- Rendered in MJCF with `group=1` and cyan RGBA; optional debug overlay adds
  highlight spheres in the viewer.

## Implementation choices

### Fixed compiled topology, sampled poses (plan fixed decision #6)

> **Retroactive backend-driven revision (Stage 3):** an earlier draft used
> `max_rungs` separate mocap rung bodies plus one mocap frame. That layout was
> abandoned because MuJoCo-Warp did not generate reliable contacts for geoms on
> secondary mocap bodies. The supported topology is below. Do **not** “correct”
> the code back to per-rung mocap bodies based on obsolete notes.

- `build_ladder_spec()` creates **one** mocap **frame** body carrying vertical
  rails, cylindrical rung collision geoms, and attachment sites.
- Rungs are **not** separate mocap bodies. MuJoCo only collides geoms parented
  directly on a mocap body; per-rung mocap bodies would render correctly but not
  contact reliably under Warp.
- Reset moves the frame mocap pose plus per-env batched `model.geom_pos` /
  `model.site_pos` for rung centres and offset latch sites;
  `nbody` / `ngeom` / `nmocap` stay constant (`nmocap == 1`).
- Inactive rungs are parked at `inactive_rung_pose` in the ladder frame via the
  same batched `geom_pos` / `site_pos` path — they do not interact with the
  active scene.

### mjlab integration

- Ladder is a separate scene entity: `entities={"ladder": make_ladder_entity_cfg(cfg)}`.
- Scene-only env: `make_ladder_scene_env_cfg()` — plane terrain, no robot, no
  actions/observations/rewards; **not** registered via `register_mjlab_task`.
- Reset sampling: `EventTermCfg(mode="reset")` → `mdp.events.reset_ladder_sample`
  → `LadderRuntime.attach` / `resample`.

### Runtime state

- `LadderTopology.from_model()` resolves prefixed names (`ladder/rung_00_geom`, …),
  frame mocap ID, geom IDs, and site IDs once at attach time.
- `LadderSampler.sample(num_envs)` returns `LadderSample` tensors on `env.device`.
- `LadderRuntime.apply()` writes batched frame `mocap_pos` / `mocap_quat` and
  per-rung local `geom_pos` / `site_pos`.
- Runtime is stored in `env.extras["ladder_runtime"]` for debug and future MDP use.

### Debug tooling

- `train_mimic/scripts/debug_climb.py --mode scene`:
  - prints active mask, rung heights, frame pose on reset/resample;
  - `Enter` = env reset (resample), `N` = resample without full env reset;
  - optional `--reset-interval`, `--no-debug-vis`, `--export-scene`.
- **macOS:** native MuJoCo viewer requires `mjpython`, not plain `python`.
- Debug overlay (`debug_vis.py`) transforms site offsets with sampled
  `rung_pos_w` / `rung_quat_w` via `mju_quat2Mat` — not `MjModel.body_mat`
  (which does not exist).

### Scene persistence

Scenes are **not** saved automatically. The compiled spec lives in memory on
`env.unwrapped.scene`. Optional export:

```bash
mjpython train_mimic/scripts/debug_climb.py --mode scene \
  --export-scene /tmp/climb_scene
# -> /tmp/climb_scene/scene.xml (+ assets/ if any)

mjpython train_mimic/scripts/debug_climb.py --mode scene \
  --export-scene /tmp/climb_scene --export-scene-zip
# -> /tmp/climb_scene.zip
```

Export writes the **fixed topology** only; sampled mocap poses are runtime state.

## Completed file map

```text
train_mimic/tasks/climbing/
├── config/
│   ├── constants.py      # Stage 0 (unchanged)
│   ├── registry.py       # Stage 0 stub (still no register_mjlab_task)
│   ├── ladder.py         # LadderConfig dataclass
│   └── scene.py          # make_ladder_scene_env_cfg, make_ladder_entity_cfg
├── ladder/
│   ├── geometry.py       # RungGeometry protocol, CylinderRungGeometry, factory
│   ├── generator.py      # build_ladder_spec, stable name helpers
│   └── state.py          # LadderTopology, LadderSampler, LadderRuntime
├── mdp/
│   └── events.py         # reset_ladder_sample
├── debug_vis.py          # attachment-site debug overlay
├── STAGE0_BASELINE.md
└── STAGE1_LADDER.md      # this file

train_mimic/scripts/debug_climb.py
tests/test_climbing_stage0.py
tests/test_climbing_stage1.py
```

## Completed steps (Stage 1 checklist)

1. `LadderConfig` with topology, spacing, friction, pose ranges, parking pose,
   and `geometry_kind="cylinder"`.
2. `RungGeometry` protocol + `CylinderRungGeometry`; unsupported kinds raise
   `ValueError`.
3. Fixed-topology MJCF: **one** mocap frame + rails, rung geoms/sites on frame
   (not per-rung mocap bodies), named attachment sites offset toward the robot,
   stable rung IDs.
4. Deterministic batched sampling: strictly increasing active heights, bounded
   spacings, inactive rungs parked off-scene, no recompile on reset.
5. `debug_climb.py --mode scene` with viewer, seed, reset/resample logging.
6. Debug overlay for active rung sites (toggle with `R` in viewer).
7. Automated tests for topology invariants, sampling, batching, and tracking
   registry unchanged.

## Validation commands

```bash
# Automated (Stage 0 + Stage 1 + tracking registry)
pytest tests/test_climbing_stage0.py tests/test_climbing_stage1.py \
  tests/test_task_registry.py tests/test_train_script.py -q

# Manual scene viewer (Linux / macOS with display)
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42

# Resample-heavy inspection
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42 --reset-interval 5

# Export compiled MJCF without opening viewer logic after export
mjpython train_mimic/scripts/debug_climb.py --mode scene --export-scene /tmp/climb_scene
```

### Manual viewer checklist

- Rungs are cylindrical; rails vertical; rungs span ladder width (Y).
- Active rung count and spacing change on `Enter` / `N`.
- No stale rungs from prior samples; inactive slots invisible.
- Sites span each rung (not all at center).
- Rails may extend above the top **active** rung (full topology height vs sampled
  active count is expected with current defaults).

## Exit criteria (Stage 1 gate)

| Criterion | Status |
|-----------|--------|
| Owner can inspect many deterministic/random samples in viewer | Yes (`debug_climb.py`, seed + reset/N) |
| Unit tests prove topology and sampling invariants | Yes (`tests/test_climbing_stage1.py`) |
| Rung geometry replaceable; currently pure cylinder | Yes (`create_rung_geometry`) |
| `General-Tracking-G1` untouched | Yes (registry tests pass) |
| No MJCF recompile per reset | Yes (frame mocap + batched geom/site pos) |
| No flat/invisible rung collision surface | Yes (cylinder only) |

## Known limitations

- No G1 robot in scene (Stage 2).
- `General-Climbing-G1` not registered with mjlab.
- Active rung count defaults to 4–10, not necessarily filling rail height.
- Native viewer on macOS requires `mjpython`; no `--viewer viser` yet (unlike
  `play.py`).
- Debug overlay uses spheres, not text labels.
- Exported MJCF does not embed per-episode sampled poses.

## Stage 2 preview

Next stage adds:

- `General-Climbing-G1` registration with canonical G1 + ladder entity;
- point-hand collision geoms and contact identity;
- `debug_climb.py --mode contacts`.

See `teleopit_climbing_task_plan.md` Stage 2 for the full gate.
