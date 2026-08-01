# Stage 4 — Observation Groups and Swappable Ladder Perception

Internal handoff note for `General-Climbing-G1`. Update when observation
groups, relative-rung selection, encoders, or model fusion change.

## Scope

Stage 4 delivers:

- separated observation groups (proprio/history, contact/latch, ladder,
  privileged critic);
- `RelativeRungProvider` with deterministic K-rung torso-frame geometry;
- vector and depth ladder encoder interfaces with shared latent width;
- `ClimbingModel` fusing proprio history + ladder latents;
- `debug_climb.py --mode observations`.

Stage 4 does **not** add rewards, terminations, or a production PPO run
(Stage 5+ / Stage 8).

## Observation architecture

```text
Actor path:
  actor_proprio (1D, current)
    ├─ joint pos rel / vel
    ├─ base ang vel / projected gravity
    ├─ prev joint + latch actions (sliced by named action terms)
    └─ hand contact / attach + attached-rung ladder-relative height
  actor_proprio_history (B, T=10, D_proprio)
  actor_ladder (B, K=6, 7)
    └─ endpoint_a(3) + endpoint_b(3) + valid(1) per slot

Critic path:
  critic_proprio (+ same terms, no corruption)
  critic_proprio_history
  critic_ladder (same geometry as actor)
  critic_privileged
    ├─ base linear velocity
    ├─ pelvis height
    ├─ ladder frame pos in torso frame
    ├─ ladder frame ori in torso frame (6D two-axis representation)
    └─ active rung ladder-relative heights `[B, K]` + explicit valid mask `[B, K]`
```

Rewards and terminations remain unwired and must continue to read simulator
task state directly in Stage 5 (plan fixed decision #4).

### Actor proprio dimension

`D_proprio = 101`:

| Term | Dim |
|------|-----|
| `robot_joint_pos_rel` | 29 |
| `robot_joint_vel` | 29 |
| `robot_base_ang_vel_b` | 3 |
| `robot_projected_gravity_b` | 3 |
| `prev_joint_action` | 29 |
| `prev_latch_action` | 2 |
| `hand_in_contact` | 2 |
| `hand_attached` | 2 |
| `attached_rung_rel_height` | 2 |

Base linear velocity is **critic-only** via `critic_privileged`.

Foot contact bits are deferred until reliable foot/rung sensing exists.

## Ladder observation mode

Config: `LadderObservationConfig` (`config/observations.py`)

| Field | Default | Notes |
|-------|---------|-------|
| `mode` | `"relative_rungs"` | `"relative_rungs"` or `"depth"` |
| `depth_height` | 64 | Reserved for future depth provider |
| `depth_width` | 64 | Reserved for future depth provider |

Behavior:

- `relative_rungs`: attach `RelativeRungProvider`; actor/critic ladder groups
  are `[B, K, 7]`.
- `depth`: **fail during env/configuration construction** with
  `NotImplementedError` until `DepthCameraProvider` exists. Do not defer the
  failure to observation capture.

Model wiring (`config/rl.py` → `ladder_encoder_cfg.mode`) selects the encoder
via `build_ladder_encoder()`:

| Mode | Encoder | Ladder input |
|------|---------|--------------|
| `relative_rungs` | `RelativeRungVectorEncoder` | `[B, K, 7]` |
| `depth` | `DepthLadderEncoder` | `[B, 1, H, W]` |

A unit test builds `ClimbingModel` with `mode="depth"` and dummy depth tensors
without editing model source.

## Relative-rung provider

Config: `RelativeRungObsConfig` (`config/observations.py`)

| Field | Default | Notes |
|-------|---------|-------|
| `num_rungs` | 6 | Fixed K |
| `rungs_below` | 2 | Active rungs at/below reference height |
| `rungs_above` | 4 | Active rungs above reference height |
| `reference_body` | `pelvis` | Window selection height |
| `torso_body` | `torso_link` | Observation frame |

Endpoint transform (unchanged):

```text
p_endpoint_torso = R_torso^T (p_endpoint_world − p_torso_world)
```

### Ladder-relative height selection

Window selection uses ladder-relative scalars, **not world Z**:

```text
h_i = u_L^T (p_i − p_L)
h_ref = u_L^T (p_ref − p_L)
```

where `u_L` is the ladder frame +Z axis in world coordinates and `p_L` is the
ladder frame origin. Endpoints are still reported in the torso frame.

Selection rules:

1. Consider **active** rungs only (`LadderRuntime.sample.active_mask`).
2. Sort active rungs by ladder-relative height ascending (stable tie-break).
3. Take up to two highest rungs with `h_i <= h_ref`.
4. Take up to four lowest rungs with `h_i > h_ref`.
5. Pad remaining slots with zeros and `valid=0`.
6. Never expose rung IDs to the actor observation tensor.

Implementation: `ladder/relative_rungs.py` (`RelativeRungProvider`), attached
at startup via `attach_relative_rung_provider`.

## Mask-safe normalization

Invalid rung slots must not become artificial geometry after normalization.

Rules:

- The `valid` bit (channel 6) is **excluded** from empirical normalization.
- Only the six continuous endpoint channels are normalized.
- After normalization, continuous channels are multiplied by the validity mask.
- `RelativeRungVectorEncoder` also zeroes invalid continuous channels before
  convolution.

Critic privileged rung heights use separate `active_rung_heights_l_rel` and
`active_rung_valid_mask` terms (each `[B, K]`). Inactive slots remain height
`0` with mask `0`; do not rely on parking-pose magic heights such as `-50`.

## Attached-rung relative height

`attached_rung_rel_height` returns ladder-relative height minus reference
height. When `attached=0` or `rung_id=-1`, the value is **zero**; never gather
heights using index `-1`.

## Previous actions

`prev_joint_action` and `prev_latch_action` slice `last_action` by named
action terms (`joint_pos`, `latch`) via `action_term_slice()`, not fixed
offsets.

## Ladder encoders and model fusion

| Module | Input | Output |
|--------|-------|--------|
| `RelativeRungVectorEncoder` | `[B, K, 7]` | `[B, 64]` |
| `DepthLadderEncoder` | `[B, 1, H, W]` | `[B, 64]` |
| `NotImplementedDepthCameraProvider` | n/a | raises at env construction when `mode="depth"` |

`ClimbingModel` (`rl/climbing_model.py`):

- 1-D groups → MLP input (normalized)
- `*_history` groups → `Conv1dEncoder` over time (proprio only)
- `*_ladder` groups → mode-selected ladder encoder (not duplicated across the
  10-frame history)

RL wiring (`config/rl.py`):

```python
obs_groups={
    "actor": ("actor_proprio", "actor_proprio_history", "actor_ladder"),
    "critic": (
        "critic_proprio",
        "critic_proprio_history",
        "critic_ladder",
        "critic_privileged",
    ),
}
ladder_encoder_cfg={"mode": "relative_rungs", ...}
```

`max_iterations=1` remains intentional until Stage 8 smoke training.

## Debug: `--mode observations`

```bash
mjpython train_mimic/scripts/debug_climb.py --mode observations --seed 42
mjpython train_mimic/scripts/debug_climb.py --mode observations --seed 42 --no-debug-vis
```

Prints:

- named observation group shapes/dtypes each step;
- selected relative-rung torso-frame endpoints;
- optional endpoint spheres in world frame (`debug_observations.py`).

Manual checklist:

1. Reset with fixed seed; note relative-rung values.
2. Translate the whole scene (robot + ladder together): relative rung coords
   should stay unchanged for pure translation + fixed orientation.
3. Move only the robot root: relative coords should change with expected sign.
4. Climb/drop pelvis through the ladder: slots should scroll through the K-window;
   top/bottom slots should show `valid=0` masks.
5. Confirm `critic_privileged` is present in env obs but absent from actor RL
   group wiring.

## File map (Stage 4)

```text
train_mimic/tasks/climbing/
├── config/
│   ├── observations.py     # group wiring + RelativeRungObsConfig + LadderObservationConfig
│   ├── env.py              # observations env cfg + startup attach + depth fail-fast
│   └── rl.py               # ClimbingModel + obs_groups + ladder_encoder_cfg.mode
├── ladder/
│   └── relative_rungs.py   # RelativeRungProvider + ladder-relative window math
├── mdp/
│   └── observations.py     # observation term functions + action_term_slice
├── rl/
│   ├── ladder_encoders.py
│   └── climbing_model.py
├── debug_observations.py
└── STAGE4_OBSERVATIONS.md

train_mimic/scripts/debug_climb.py   # --mode observations
tests/test_climbing_stage4.py
```

## Validation commands

```bash
# Stage 0–4 + tracking registry
pytest tests/test_climbing_stage0.py tests/test_climbing_stage1.py \
  tests/test_climbing_stage2.py tests/test_climbing_stage3.py \
  tests/test_climbing_stage4.py tests/test_task_registry.py \
  tests/test_train_script.py tests/test_save_onnx.py -q

# Manual observation viewer
mjpython train_mimic/scripts/debug_climb.py --mode observations --seed 42
```

## Exit criteria (Stage 4 gate)

| Criterion | Status |
|-----------|--------|
| Separated proprio / ladder / privileged groups | Yes |
| Actor proprio includes contact/latch + prev actions | Yes |
| Base linear velocity critic-only | Yes |
| Relative-rung K-window with ladder-relative height selection | Yes |
| Mask-safe ladder normalization + encoder masking | Yes |
| Critic rung heights expose explicit valid mask | Yes |
| `ladder_observation.mode` selects provider/encoder | Yes |
| Depth mode fails at env construction | Yes |
| Depth-mode model forward test with dummy `[B,1,H,W]` | Yes |
| Vector + depth encoder shared latent width | Yes |
| `ClimbingModel` fuses history + ladder | Yes |
| Actor cannot consume critic-only groups | Yes |
| `debug_climb.py --mode observations` | Yes |
| Rewards independent of observation tensors | Yes (still unwired) |
| `General-Tracking-G1` untouched | Yes |

## Known limitations

- No production `DepthCameraProvider`; depth path is encoder + config wiring only.
- Foot contact bits not exposed yet.
- RL cfg registers `ClimbingModel` but remains non-trainable (`max_iterations=1`).
- Observation debug overlay uses endpoint spheres, not line segments.

## Stage 5 preview

Next: reward terms, event memory, terminations, metrics, and
`debug_climb.py --mode rewards`. Keep reward arithmetic on simulator task state,
not observation tensors.
