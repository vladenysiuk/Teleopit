# Stage 6 — Assisted contact-pose validation (`assisted_hold`)

Internal handoff note for `General-Climbing-G1`. Update when hold-pose
geometry, PD stiffening, or debug tooling change. **Thresholds in
`config/hold_pose.py` are frozen** — do not drift without owner approval.

## Scope

Stage 6 delivers **assisted contact-pose validation**, not sustained
four-contact ladder standing with training-parity gains:

- mink least-squares IK for a four-contact ladder pose;
- one-shot pose write + force-seeded dual-hand latches (no per-step teleport);
- PD joint hold under gravity with metrics + failure classification;
- `debug_climb.py --mode hold-pose`;
- saved debug artifact `data/hold_pose_seed42.npz`.

An assisted initialization produces a finite, briefly supported ladder pose.
Sustained four-contact standing with training-parity gains is deferred to
learnability experiments (Stage 9+).

Stage 6 does **not** add scripted MDP agents, reset-stress suites, or PPO
smoke training (Stage 7+ / Stage 8).

## Interpretation

| Observation (seed 42) | Meaning |
|-----------------------|---------|
| `max_pelvis_drop_m = 0.13489246368408203` | ~13.5 cm pelvis motion under gravity |
| `max_foot_slip_m = 0.07081150263547897` | ~7.1 cm tangential foot crawl on cylinders |
| `mean_foot_contact_fraction = 0.7262396694214877` | intermittent dual-foot support |
| `any_foot_contact_fraction = 1.0` | at least one foot always in contact |

**0.135 m pelvis motion and ~0.071 m foot crawl do not establish stable
standing.** They only show brief assisted support under stiffened hold-env PD.

Stricter targets for future learnability diagnostics are recorded in
`FUTURE_DIAGNOSTIC_GOALS` (`config/hold_pose.py`) and are **not** Stage 6
blockers:

| Future diagnostic goal | Target |
|------------------------|--------|
| Max pelvis drop | ≤ 0.10 m |
| Max tangential foot slip | ≤ 0.05 m |
| Mean both-feet contact | ≥ 0.80 |
| Max hand attach error | ≤ 0.02 m |
| Leg saturation fraction | ≤ 0.50 |

## Stage 3 lessons carried forward

| Pitfall | Stage 6 handling |
|---------|------------------|
| Per-step `write_root_link_pose_to_sim` ignores collision | Pose written **once** at init; hold uses PD actions only |
| Latch attach needs previous-step contact | Force-seed connect equalities after pose place |
| Latch sites are external hand-sphere centres | IK targets latch sites, not rung axes |
| Viewer needs `expand_model_fields(geom_pos, site_pos)` | Inherited from `LadderRuntime.attach` |
| Do not revert to per-rung mocap bodies | Unchanged frame-parented geometry |
| Warp contacts ≠ `mj_data.ncon` | Metrics use `sim.data.nacon` / sensors / geom distance |

## Hold pose contract (`assisted_hold`)

Default assignment on the fixed 6-rung debug ladder:

| Contact | Rung | Site / Y |
|---------|------|----------|
| Left foot | 1 | `y = -0.08` m |
| Right foot | 1 | `y = +0.08` m |
| Left hand | 3 | site 1 |
| Right hand | 3 | site 3 |

Allowed Stage-6 tuning (no forbidden shortcuts):

- rung radius `0.035` m;
- higher cylinder / foot sliding friction;
- PD stiffness ×8 / damping ×6 vs stock G1 training actuators;
- slightly stiffer latch `solref=(0.012, 1.0)`.

**Not used:** flat/invisible supports, unlimited torques, gravity off, foot
welds, mass changes.

## Debug: `--mode hold-pose`

```bash
mjpython train_mimic/scripts/debug_climb.py --mode hold-pose --seed 42
mjpython train_mimic/scripts/debug_climb.py --mode hold-pose --seed 42 \
  --hold-duration 2.5 --save-hold-npz /tmp/hold_pose.npz
```

### Manual checklist

1. Feet rest on pure cylinders (no flat boxes / foot welds).
2. Hands stay attached to the intended rung sites.
3. Robot is supported (not floating); some weight through feet.
4. Pelvis motion and foot slip are recorded numerically — not judged by video alone.
5. Failure class / metrics are printed.

## Acceptance metrics (seed 42, 2.5 s, frozen `assisted_hold` gates)

| Metric | Gate | Observed (seed 42) |
|--------|------|--------------------|
| Duration | ≥ 2 s | 2.42 s (`steps=484`, `dt=0.005`) |
| Max pelvis drop | ≤ 0.16 m | 0.13489246368408203 m |
| Max tangential foot slip | ≤ 0.18 m | 0.07081150263547897 m |
| Mean both-feet contact | ≥ 0.40 | 0.7262396694214877 |
| Any-foot contact | ≥ 0.80 | 1.0 |
| Max hand attach error | ≤ 0.04 m | 0.004356163553893566 m |
| Leg saturation fraction | ≤ 0.85 | 0.0 |
| Nonfinite / capacity overflow | none | `failure=ok`, `max_nacon=28`, `max_nefc=118` |
| Spawn sole radial | > rung radius | clear on seeds 42 and 45455 |

Failure classification runs before further parameter changes
(`pelvis_drop`, `foot_slip`, `lost_foot_contact`, `latch_compliance`, …).

## File map (Stage 6)

```text
train_mimic/tasks/climbing/
├── config/
│   ├── hold_pose.py        # HoldPoseConfig + FUTURE_DIAGNOSTIC_GOALS
│   └── env.py              # make_climbing_hold_pose_env_cfg
├── ladder/hold_ik.py       # mink IK + world targets
├── debug_hold_pose.py      # init, PD hold action, metrics, classifier
├── data/hold_pose_seed42.npz
└── STAGE6_HOLD_POSE.md

train_mimic/scripts/debug_climb.py   # --mode hold-pose
tests/test_climbing_stage6.py
```

## Validation commands

```bash
pytest tests/test_climbing_stage0.py tests/test_climbing_stage6.py \
  tests/test_climbing_stage7.py::test_scripted_hold_reproduces_stage6_thresholds -q

mjpython train_mimic/scripts/debug_climb.py --mode hold-pose --seed 42
```

## Exit criteria (Stage 6 gate)

| Criterion | Status |
|-----------|--------|
| Deterministic IK pose finite / within joint limits | Yes |
| Expected hand/rung and foot/rung pairs at init | Yes |
| No gross penetration at initialization | Yes |
| Finite sim for requested duration | Yes |
| No constraint/contact buffer overflow | Yes |
| Metrics logger expected shapes | Yes |
| ≥2 s assisted hold with recorded metrics | Yes (2.42 s, seed 42) |
| No forbidden physical shortcuts | Yes |
| `debug_climb.py --mode hold-pose` | Yes |
| `General-Tracking-G1` untouched | Yes |

## Known limitations

- Hold env uses stiffer PD than training actuators; this validates
  `assisted_hold`, not training-parity standing.
- Saved NPZ is debug/test data, not a motion clip for imitation.
- Detailed force/torque diagnostics deferred to Stage 8/9 unless cheap via existing APIs.
- RL cfg still not trainable until Stage 8.

## Stage 7 preview

Next: reset robustness, zero/random/scripted MDP agents, batch capacity checks,
and end-to-end MDP validation without PPO.
