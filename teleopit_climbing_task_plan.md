# TeleopIt G1 Ladder-Climbing Task: Staged Implementation Plan

## Purpose

Implement a new `General-Climbing-G1` task alongside TeleopIt's existing
tracking task. Reuse TeleopIt's mjlab/rsl_rl training infrastructure, canonical
G1 model, joint-position control, logging, and runner. Add:

- a procedurally sampled ladder;
- point-hand contact geoms;
- attach/detach actions and latches;
- proprioception plus privileged relative-rung observations;
- a handwritten climbing reward and climbing-specific terminations;
- deterministic validation tools that do not require a trained policy.

This plan is intentionally divided into gated stages. The implementation agent
must stop after each stage and wait for the owner to validate it. Do not combine
several stages into a single large change.

## Fixed design decisions

These are requirements, not open design questions:

1. **Rungs initially use pure cylindrical collision geometry with high
   friction.** Do not add an invisible flat collision surface. Geometry must
   nevertheless be selected through a small factory/configuration layer so a
   different rung profile can be substituted later.
2. **The first policy receives privileged relative rung geometry.** Use exact
   simulator rung positions expressed in a robot-relative frame.
3. **Observation architecture must admit a future raw-depth provider.** Keep
   proprioception and ladder exteroception as separate groups/encoders before
   fusion. Implement the relative-rung path now; a production depth-camera path
   is not part of the initial task.
4. **Rewards must consume simulator task state, not the actor observation.**
   Changing the perception provider later must not change reward semantics.
5. **Create a new climbing task.** Do not turn `General-Tracking-G1` into a
   climbing task and do not break its observation/action/ONNX contract.
6. **Use a fixed compiled maximum rung count.** Sample poses and active masks at
   reset; do not recompile MJCF for every episode. Ladder topology is **one**
   mocap frame body with batched rung `geom_pos` / `site_pos` (not per-rung
   mocap bodies); Warp contacts on secondary mocap bodies were unreliable.
7. **Do not begin a serious PPO run until deterministic reward, contact, latch,
   observation, and static-hold validations pass.**
8. **Latch attachment sites target the external hand-sphere centre**, not the
   rung centreline:
   `p_target = p_rung_axis − (r_rung + r_hand + δ) e_x_ladder`
   (robot approaches from −ladder X). Keep hand–rung collision enabled while
   attached; do not pull the sphere into the cylinder axis.

## General instructions to the implementation agent

Before every stage:

- Read the repository's `AGENTS.md` and any more local instructions.
- Inspect `git status`; preserve all unrelated owner changes.
- Adapt provisional filenames below to current repository conventions, but
  explain any deviation.
- Keep the change scoped to the current stage.
- Add or update tests in the same stage as the implementation.
- Run the narrow tests first, then relevant existing tests.
- Do not commit, push, open a PR, or modify external services unless explicitly
  asked.

At the end of every stage, report:

1. files changed;
2. architectural choices made;
3. exact commands run;
4. test results;
5. manual validation command;
6. what the owner should observe;
7. known limitations;
8. whether the stage's exit criteria are satisfied.

If a stage is blocked, stop and report evidence. Do not silently replace a
required mechanism with a materially different approximation.

## Suggested package layout

Treat this as provisional and align it with current TeleopIt/mjlab conventions:

```text
train_mimic/tasks/climbing/
├── __init__.py
├── config/
│   ├── constants.py
│   ├── env.py
│   ├── ladder.py
│   ├── registry.py
│   └── rl.py
├── mdp/
│   ├── actions.py
│   ├── events.py
│   ├── observations.py
│   ├── rewards.py
│   └── terminations.py
├── rl/
│   ├── ladder_encoders.py
│   └── climbing_model.py
├── ladder/
│   ├── geometry.py
│   ├── generator.py
│   └── state.py
└── tests/
```

Add one reusable debug entry point:

```text
train_mimic/scripts/debug_climb.py
```

It should grow stage by stage and eventually support:

```bash
python train_mimic/scripts/debug_climb.py --mode scene
python train_mimic/scripts/debug_climb.py --mode contacts
python train_mimic/scripts/debug_climb.py --mode latch
python train_mimic/scripts/debug_climb.py --mode observations
python train_mimic/scripts/debug_climb.py --mode rewards
python train_mimic/scripts/debug_climb.py --mode hold-pose
```

---

# Stage 0 — Baseline, dependency pinning, and implementation contract

## Agent tasks

1. Inspect:
   - `AGENTS.md`;
   - `pyproject.toml` and resolved versions of MuJoCo, MuJoCo-Warp, mjlab,
     rsl_rl, PyTorch, and CUDA;
   - the current `General-Tracking-G1` registry, environment builder, robot
     spec loader, observation model, action manager, and training entry point.
2. Confirm the canonical G1 XML is available. Do not substitute a different G1
   asset.
3. Run an existing TeleopIt import check and the smallest available existing
   play/smoke test.
4. Write a short internal implementation note, preferably in the climbing
   package or test documentation, recording:
   - relevant versions;
   - existing G1 joint action dimension;
   - policy and physics rates;
   - wrist body names and point offsets;
   - existing task registration pattern;
   - how custom sensors/actions/rewards access batched simulation data.
5. Create only the empty climbing package/registry skeleton required to prove it
   imports. Do not add ladder physics yet.

## Automated checks

- `import train_mimic.tasks` still succeeds.
- Existing tracking task registration is unchanged.
- New climbing package imports without registering a broken environment.
- Existing narrow/fast tests still pass.

## Owner validation

Ask the agent to show:

- dependency/version table;
- clean baseline command and output;
- a concise proposed file map;
- confirmation that the tracking task is untouched.

No visual simulation is necessary at this stage.

## Exit criteria

- Baseline works before climbing changes.
- Exact backend versions are recorded.
- The agent can identify the current extension points without guessing.

---

# Stage 1 — Ladder specification, generation, and manual viewer

## Agent tasks

1. Implement a configuration dataclass containing at least:
   - `max_rungs`;
   - active rung-count range;
   - rung radius and length;
   - rung friction tuple;
   - base height;
   - spacing range;
   - ladder distance, yaw/tilt ranges;
   - rail geometry;
   - number of attachment sites per rung;
   - inactive-rung parking pose;
   - `geometry_kind="cylinder"`.
2. Implement a rung-geometry factory/protocol. Only
   `CylinderRungGeometry` must work now. An unsupported geometry kind must fail
   clearly rather than silently falling back.
3. Build a fixed-topology ladder spec:
   - one mocap frame body carrying all rung collision geoms and sites
     (not per-rung mocap bodies);
   - pure cylinder collision geom per rung;
   - named attachment sites distributed along each rung and offset toward the
     robot by `(r_rung + r_hand + δ)` from the rung axis;
   - stable IDs independent of sampled height;
   - rails as simple visual/collision geometry;
   - per-env layout via batched `geom_pos` / `site_pos` plus frame mocap pose.
4. Implement deterministic per-environment sampling:
   - strictly increasing rung heights;
   - bounded spacings;
   - fixed maximum topology;
   - inactive rungs moved outside the usable scene and unable to interfere;
   - no MJCF recompilation during reset.
5. Implement `debug_climb.py --mode scene`:
   - one environment;
   - fixed seed;
   - viewer;
   - optional reset key/interval;
   - display or log rung IDs, heights, active mask, and sampled ladder pose.
6. If viewer debug drawing is available, label rung IDs or show attachment
   sites.

## Automated checks

- Same seed produces identical ladder state.
- Different seeds produce different valid ladders.
- Active rung count is within bounds.
- Active heights are strictly increasing.
- Every adjacent spacing is within configured bounds.
- Rung geoms are cylinders with expected radius, length, and friction.
- Rung names, IDs, and attachment-site ordering are stable.
- Inactive rungs are outside the active scene and never selected.
- Reset changes only per-environment state; compiled topology counts are stable.
- Sampling works for a batch, not only environment zero.

## Owner validation

Run the scene viewer repeatedly. Confirm visually:

- rungs are genuinely cylindrical;
- rails and rungs have the intended orientation;
- spacing/count/distance change on reset;
- no rung visibly remains from the previous sample;
- inactive rungs do not appear;
- sites span the rung rather than all lying at its centre;
- the ladder is sensibly scaled relative to the G1.

Suggested initial scales for inspection, not immutable constants:

- radius: roughly 25–35 mm;
- vertical spacing: roughly 220–280 mm;
- width compatible with the G1 stance and arm span;
- deliberately high initial sliding friction.

Do not approve a flattened collision surface at this stage.

## Exit criteria

- The owner can inspect many deterministic/random samples in a viewer.
- Unit tests prove topology and sampling invariants.
- Rung geometry is easily replaceable in code but currently pure cylinder.

---

# Stage 2 — Climbing task skeleton, G1 integration, and point-hand contacts

## Agent tasks

1. Register a separate `General-Climbing-G1` environment.
2. Reuse the canonical G1 robot spec and existing joint-position actuator
   configuration. Do not copy the entire G1 XML into a new divergent asset.
3. Add to the G1 training spec:
   - `left_hand_point` and `right_hand_point` spherical collision geoms;
   - colocated named attachment sites;
   - the previously agreed wrist-relative offsets unless inspection shows the
     canonical model uses a different frame.
4. Configure collision filtering so:
   - point hands collide with ladder rungs;
   - feet and relevant robot collision geoms can still collide with rungs;
   - point-hand geoms do not introduce unintended self-collisions.
5. Add contact sensing for each hand against active ladder rungs.
6. Implement a `ClimbState`/contact adapter returning batched:
   - `hand_in_contact[B,2]`;
   - `hand_rung_id[B,2]`, with `-1` for none;
   - `hand_rung_height[B,2]`;
   - optional contact position and force for debugging.
7. Define deterministic multiple-contact selection:
   - prefer the matching rung with maximum normal force;
   - document and test tie behaviour.
8. Implement `--mode contacts`. Avoid requiring IK for this test: place a
   selected mocap rung relative to the current wrist so it can be moved from
   clear separation through contact.
9. Visualize:
   - point-hand spheres;
   - contacted rung;
   - contact point/force if supported;
   - live contact flag, rung ID, and height.

## Automated checks

- Far hand: no contact, ID `-1`.
- Near but separated hand: no false contact beyond configured margin.
- Touching/slightly penetrating hand: correct contact and positive force.
- Correct rung ID and sampled height are returned.
- Left/right hands are independent.
- Contact with an inactive rung is impossible.
- Multiple-contact reduction follows the documented rule.
- Reset clears stale contact/history state.
- Batched environments resolve their own rung states independently.
- Observation/contact tensors have expected dtype, device, and shape.

## Owner validation

In the contact viewer:

1. Move a rung away from the left hand: contact must be false.
2. Move it into contact: the correct rung must highlight and print its height.
3. Move it away: contact must clear promptly.
4. Repeat for the right hand.
5. Place both in contact simultaneously.
6. Pause/single-step near the contact boundary and watch for flicker or stale
   states.

The robot does not need to attach or stand yet.

## Exit criteria

- Contact identity is correct, not just a generic boolean.
- Contact state resets correctly and works in a batch.
- Point-hand geometry creates no obvious unintended robot collisions.

---

# Stage 3 — Attach/detach action and latch backend

## Agent tasks

1. Add a two-dimensional latch action after the existing joint actions.
2. Implement hysteresis:
   - `g > attach_threshold`: request attach;
   - `g < detach_threshold`: request detach;
   - values between thresholds preserve current state.
3. Implement per-environment, per-hand latch state:
   - detached/attached;
   - attached rung ID;
   - attached site ID;
   - transition/event flags;
   - reset.
4. Disallow direct switching from rung `i` to rung `j`; require detach first.
5. Predeclare inactive site-based `connect` constraints for valid
   hand/rung/site combinations. On attach:
   - require current valid contact/capture condition;
   - select the closest eligible attachment site;
   - activate exactly one constraint for that hand.
6. On detach, deactivate the constraint immediately.
7. Keep the latch implementation behind a small `LatchBackend` interface.
8. Verify whether the pinned MuJoCo-Warp backend supports per-world dynamic
   equality activation. If it does not:
   - stop;
   - provide a minimal reproducer and backend evidence;
   - propose a spring-damper backend;
   - wait for owner approval before changing mechanisms.
9. Add optional configuration for capture radius and future break force. A
   force-limit feature may remain disabled if reliable constraint-force access
   is not yet available.
10. Implement `--mode latch` with deterministic attach, pull, and detach
    sequences.

## Automated checks

Test the full transition table:

| Initial | Contact | Command | Expected |
|---|---:|---:|---|
| detached | no | attach | detached + invalid request |
| detached | yes | neutral/detach | detached |
| detached | yes | attach | attached to contacted rung |
| attached | any | attach/neutral | remains on same rung |
| attached | any | detach | detached |
| attached to `i` | touches `j` | attach | remains on `i` |

Also test:

- exactly one active constraint per hand;
- both hands can attach independently;
- nearest site is selected;
- attachment error remains bounded during a pull test;
- release removes the constraint;
- hysteresis prevents action-threshold chatter;
- reset clears all equality and Python-side state (deactivate equalities
  before resampling ladder sites);
- batched per-world isolation (attach different hands/rungs in ≥4 envs;
  subset reset clears only intended worlds);
- during attach/pull: bounded penetration, contact normal force, and equality
  residual consistent with the external latch target;
- CPU MuJoCo and one-world MuJoCo-Warp agree qualitatively;
- no NaNs or constraint-capacity overflow.

## Owner validation

Watch a scripted sequence:

1. hand touches rung while latch command is neutral;
2. attach command activates latch;
3. robot/rung is perturbed and the hand remains connected;
4. touching another rung does not silently switch attachment;
5. detach command releases it;
6. attach request in free space fails;
7. left and right latches work simultaneously;
8. reset removes both.

Inspect the live list of active constraint names/IDs. The visual connection must
match the reported rung.

## Exit criteria

- Latch transitions are deterministic and independently batched.
- The exact backend mechanism is demonstrated on the pinned GPU stack.
- No fallback approximation has been introduced without approval.

---

# Stage 4 — Observation groups and swappable ladder-perception architecture

## Agent tasks

1. Define separate observation groups:
   - proprioception/history;
   - contact/latch state;
   - ladder exteroception;
   - privileged critic state.
2. Initial deployable actor proprioception should include:
   - joint position relative to nominal;
   - joint velocity;
   - base angular velocity;
   - projected gravity;
   - previous 29 joint actions and two latch actions;
   - hand contact and attachment bits;
   - attached-rung relative height or equivalent compact state;
   - optionally foot contact bits if reliably available.
3. Keep base linear velocity and other nontrivially observable simulator state
   in the critic group unless deliberately exposed as a noisy estimator.
4. Implement `RelativeRungProvider`:
   - select a fixed `K`, initially six;
   - choose a deterministic local window using **ladder-relative height**
     `h = u_L^T (p - p_L)`, not world Z (two rungs below and four above pelvis
     ladder-relative height);
   - provide both rung endpoints in the torso frame plus a validity mask;
   - never expose unstable absolute rung IDs as semantic inputs;
   - handle fewer than `K` active rungs with masks and finite padding.
5. Use the transform

   ```text
   p_endpoint_torso = R_world_torso^T
                      (p_endpoint_world - p_torso_world)
   ```

6. Keep rewards and terminations independent of these observation tensors.
7. Implement separate ladder encoders:
   - working vector encoder for `[B,K,7]` or its documented equivalent;
   - a depth-encoder interface accepting `[B,1,H,W]`;
   - both return the same configured ladder latent width.
8. Add `ladder_observation.mode = "relative_rungs" | "depth"`. The model builder
   must select provider/encoder from that field. Depth mode must fail during
   environment/configuration construction until a real `DepthCameraProvider`
   exists. Include a test that swaps in dummy `[B,1,H,W]` depth input and runs a
   policy forward pass via configuration only.
9. Make padding and normalization mask-safe: exclude the valid bit from
   continuous normalization; zero invalid geometry after normalization or inside
   the encoder. Critic privileged rung heights must carry an explicit validity
   mask, not magic inactive heights.
10. Fuse proprioceptive and ladder latents in a climbing-specific actor/critic
    while reusing the existing PPO runner.
11. Preserve temporal history for proprioception. Do not blindly duplicate a
    large image across the existing 10-frame history.
12. Implement `--mode observations` to print named shapes and selected relative
    rung geometry, with optional debug lines from torso to endpoints.

Additional checks:

- detached `attached_rung_relative_height` returns zero with `attached=0`;
- slice previous joint/latch actions by named action terms;
- privileged ladder orientation uses 6D/two-axis representation or
  canonicalized quaternions;
- confirm the actor cannot consume critic-only groups.

## Automated checks

- Exact known transforms give exact expected torso-frame endpoints.
- Translating robot and ladder together leaves relative observations unchanged.
- Rotating robot and ladder together leaves torso-frame observations unchanged.
- Rung ordering is deterministic using ladder-relative height, including on
  tilted ladders.
- Masks and padding are correct near ladder top/bottom and survive normalization.
- Inactive rungs are never exposed as valid.
- All observation values are finite after reset and before termination.
- Observation tensors have expected batch/device/dtype/shape.
- Vector encoder returns configured latent size.
- Dummy depth encoder/model path returns the same latent size for representative
  image shapes via `ladder_observation.mode="depth"`.
- Depth mode fails at env construction when the camera provider is absent.
- Critic may use privileged state; actor does not accidentally receive it.
- Existing tracking model/export tests remain unaffected.

## Owner validation

Use a fixed ladder and move/rotate the whole scene:

- printed relative rung coordinates should remain unchanged under common rigid
  motion;
- moving only the robot should change them with the expected sign;
- selected rungs should move smoothly through the `K`-rung window;
- masks should appear at the top and bottom;
- debug lines should terminate at the visually correct rung endpoints.

Ask the agent to show a model/observation diagram and both encoder signatures.

## Exit criteria

- Privileged relative-rung observations are correct and stable.
- Ladder modality is separated from proprioception before fusion.
- A later depth path has a real architectural insertion point without affecting
  reward or task state.

---

# Stage 5 — Reward terms, event memory, metrics, and terminations

## Agent tasks

1. Implement each reward as a small independently testable function returning
   shape `[B]`.
2. Keep every weight in configuration; no hidden numeric weights in functions.
3. Implement at least:

   ```text
   + pelvis/torso upward progress
   + one-off new higher attachment
   + success
   - physical-time penalty
   - action-rate penalty
   - effort/torque penalty
   - invalid latch request
   - optional latch overload
   ```

4. Use pelvis or torso height for climbing progress, never hand height.
5. Normalize time penalty by physical duration:

   ```text
   r_time = -c_time * dt * not_success
   ```

6. Prevent attachment reward farming:
   - maintaining an attachment receives no new event reward;
   - detach/reattach to the same or lower rung does not repeatedly pay;
   - per-environment event memory resets correctly.
7. Define success clearly, for example pelvis/torso above a target height while
   maintaining a valid support condition. Make the exact predicate configurable
   and log it as a metric.
8. Implement terminations:
   - success;
   - fall/base too low or too far from ladder;
   - timeout;
   - numerical invalidity if supported;
   - optional latch overload.
9. Add per-term logging and episode metrics:
   - maximum pelvis height;
   - number of valid higher attachments;
   - invalid latch count;
   - success;
   - foot/hand contact fractions;
   - time to success;
   - torque saturation fraction.
10. Implement `--mode rewards` with deterministic synthetic transitions and a
    short scripted simulator rollout.

## Automated checks

Use a table-driven reward oracle:

- unchanged state: only expected time/regularization cost;
- pelvis rises exactly 0.1 m: exact progress reward;
- hand rises alone: zero progress reward;
- first higher attachment: one event reward;
- persistent attachment: no repeated reward;
- detach/reattach same rung: no farming;
- invalid free-space attach: exact penalty;
- known action difference: exact action-rate cost;
- one physical second: exact total time cost independent of control rate;
- success: exact bonus and termination;
- global robot+ladder translation: reward invariant;
- left/right mirrored state: reward invariant when configuration is symmetric;
- all term outputs have shape `[B]` on the correct device.

Integration-test the configured reward manager against an independent manual
sum for known states.

## Owner validation

Inspect live per-term plots during a scripted sequence:

1. raise a hand only;
2. attach to a higher rung;
3. wait;
4. raise pelvis;
5. make an invalid latch request;
6. trigger success.

Expected:

- hand raising alone gives no progress;
- attachment bonus fires once;
- waiting accumulates time cost;
- pelvis motion gives progress;
- invalid request gives its penalty;
- success fires once and terminates.

Review the magnitude of every term. No routine penalty should dominate useful
progress by orders of magnitude.

## Exit criteria

- Unit tests independently validate arithmetic and event semantics.
- Plots show the intended terms at the intended transitions.
- Time pressure distinguishes efficient climbing from unnecessary motion.

---

# Stage 6 — Static IK pose and brief ladder-standing validation

## Agent tasks

1. Implement or reuse a simple IK/least-squares pose generator targeting:
   - two feet on cylindrical rung tops;
   - two hands at higher rung sites;
   - upright torso near ladder;
   - joint-limit regularization;
   - minimal obvious self/ladder penetration.
2. Save the deterministic pose as test/debug data only after it is validated.
3. Implement `--mode hold-pose`:
   - one environment;
   - no randomization;
   - initialize exact pose and zero velocity;
   - attach both hands;
   - command the existing PD joint controller to hold the pose;
   - simulate under gravity for 2–5 seconds.
4. Record/plot:
   - pelvis position and drop;
   - hand attachment error and load if available;
   - per-foot contact and rung ID;
   - tangential foot slip;
   - joint torque and saturation;
   - total contact vertical force;
   - constraint/contact counts;
   - NaN/solver/capacity warnings.
5. If the hold fails, classify the cause before changing parameters:
   - foot slip;
   - foot rolling off;
   - infeasible IK;
   - torque saturation;
   - poor PD gains;
   - latch compliance;
   - collision/initial penetration;
   - solver capacity.
6. Allowed tuning within this stage:
   - cylinder friction;
   - cylinder radius within the agreed range;
   - initial pose;
   - PD gains/limits if physically/model-consistent;
   - solver/contact parameters and capacities.
7. Not allowed without owner approval:
   - flattened/invisible support surfaces;
   - unlimited actuator torques;
   - disabling gravity;
   - welding feet to rungs;
   - changing robot mass;
   - treating the static hold as proof of real-world feasibility.
8. After the four-contact hold works, optionally add a short scripted
   one-foot-unload test: shift support, lift one foot briefly, and restore it.

## Automated checks

At minimum:

- deterministic IK pose is finite and within joint limits;
- expected hand/rung and foot/rung pairs are initially correct;
- no gross penetration at initialization;
- simulation remains finite for the requested duration;
- no constraint/contact buffer overflow;
- data logger returns expected shapes.

Suggested provisional hold acceptance thresholds:

- holds at least 2 seconds;
- pelvis drop below 10 cm;
- each foot slips less than 5 cm;
- both feet retain contact for most of the interval;
- no persistent extreme penetration;
- major leg joints are not all continuously saturated.

These thresholds are sanity checks, not research claims.

## Owner validation

Watch the hold in real time and slow motion. Confirm:

- feet rest on pure cylinders;
- the robot is supported rather than floating;
- hands are attached to the intended rungs;
- the feet do not secretly have welds or flat support boxes;
- the robot does not hang almost entirely from unlimited hand constraints;
- plotted torques and forces are plausible enough to proceed.

If it fails, ask for plots and a failure classification before permitting
parameter changes.

## Exit criteria

- At least one plausible four-contact pose survives briefly under gravity.
- Failure/success metrics are recorded, not judged solely by video.
- No forbidden physical shortcut was introduced.

---

# Stage 7 — Reset robustness, scripted agents, and end-to-end MDP checks

## Agent tasks

1. Exercise repeated randomized resets with all state enabled:
   - ladder sampling;
   - contact histories;
   - latch state/equalities;
   - reward event memory;
   - observation history;
   - previous actions;
   - success/fall/timeout state.
2. Add a deterministic scripted agent that can:
   - hold the validated pose;
   - attach/detach on command;
   - make deliberate invalid requests;
   - trigger a simple termination.
3. Run zero-action and random-action agents using existing mjlab support.
4. Verify reward/termination/metric logging through whole episodes.
5. Check simulation capacities:
   - maximum contacts;
   - maximum constraints;
   - no dropped contacts;
   - no `nconmax`/`njmax` overflow.
6. Run at increasing batch sizes, e.g. 1, 64, and a moderate GPU batch, without
   tuning for final throughput yet.
7. Add a reproducibility test for fixed seed and scripted actions.

## Automated checks

- Hundreds/thousands of resets remain finite.
- No stale latch survives reset.
- No stale "already rewarded" rung survives reset.
- All observations are finite on the first post-reset step.
- Every termination resets only intended environments.
- Fixed seed + fixed actions produces repeatable high-level results.
- Zero/random/scripted agents run without shape or manager errors.
- Existing tracking tests remain green.

## Owner validation

Run:

- zero agent: should not crash; likely falls or times out predictably;
- random agent: should reset safely and never activate impossible latches;
- scripted hold agent: should reproduce Stage 6;
- repeated reset viewer: every environment state should visibly refresh.

Ask for per-term reward summaries and termination counts, not only a video.

## Exit criteria

- The complete MDP runs safely without PPO.
- Reset and batching bugs have been ruled out at useful scale.

---

# Stage 8 — Tiny PPO smoke test and reward-scale diagnosis

## Agent tasks

1. Add climbing-specific PPO/model configuration while reusing the existing
   TeleopIt runner and logging path.
2. Run only a smoke test first, for example:
   - 64 environments;
   - roughly 100 iterations;
   - fixed easy ladder;
   - conservative initial state;
   - no large domain randomization.
3. The objective is infrastructure validation, not climbing success.
4. Log:
   - every reward term;
   - observation normalization statistics;
   - action distributions, including latch actions;
   - episode length/termination type;
   - maximum pelvis height;
   - valid/invalid attachment rates;
   - torque saturation;
   - NaN/Inf guards;
   - throughput and memory.
5. Compare reward magnitudes and revise only configurable weights if a clear
   scale pathology is demonstrated.

## Automated checks

- Training starts and checkpoints.
- Losses, observations, actions, and rewards remain finite.
- More than one episode/reset occurs.
- Reward variance is nonzero.
- Latch action dimensions are trained and logged.
- No reward term overwhelms all others unintentionally.
- Playback can load the smoke-test checkpoint.

## Owner validation

Do not judge policy quality from 100 iterations. Validate:

- curves are finite;
- environment resets are healthy;
- reward decomposition is sensible;
- action distributions are not immediately saturated;
- a checkpoint plays back;
- no unexplained warnings are ignored.

## Exit criteria

- End-to-end PPO integration is numerically stable.
- Logging is sufficient to diagnose the first real experiment.

---

# Stage 9 — First learnability experiment and curriculum hooks

## Agent tasks

1. Implement curriculum parameters as configuration, not hard-coded stage
   switches:
   - initial hand attachment probability;
   - rung spacing/count ranges;
   - ladder distance/tilt;
   - friction/radius randomization;
   - initial pose noise;
   - latch capture radius/optional force limit.
2. Start with an easy experiment:
   - fixed vertical ladder;
   - regular spacing;
   - validated near-ladder initial pose;
   - both hands attached or very easy first contact;
   - high but finite cylinder friction;
   - modest episode length.
3. Define pre-run success evidence:
   - better maximum pelvis height than zero/random baselines;
   - increased valid higher-attachment count;
   - reduced invalid latch rate;
   - some stable support transitions;
   - no growth in torque saturation/contact instability.
4. Train only long enough to test whether the signal improves these metrics.
5. Do not enable broad randomization until the fixed-ladder task shows
   learnability.

## Owner validation

Compare trained policy against zero and random baselines on the same fixed
seeds. Inspect:

- videos from several seeds, not one cherry-picked episode;
- maximum pelvis-height distribution;
- attachment sequence;
- foot contacts and slip;
- time efficiency;
- torque/latch loads;
- failure modes.

Approve curriculum expansion one axis at a time:

1. remove guaranteed initial attachment;
2. randomize spacing;
3. randomize rung count;
4. randomize distance and initial pose;
5. randomize tilt;
6. randomize radius/friction;
7. introduce missing/offset rungs if desired.

## Exit criteria

- A measurable learning signal exists on the easiest task.
- Curriculum controls are available but not prematurely activated.

---

# Stage 10 — Documentation, regression suite, and handoff

## Agent tasks

1. Document:
   - installation/assets;
   - task registration;
   - all debug modes;
   - test commands;
   - ladder and reward configuration;
   - observation definitions and frames;
   - action ordering and latch thresholds;
   - curriculum controls;
   - known backend limitations;
   - current lack of a production depth provider;
   - difference between simulation latch feasibility and real G1 capability.
2. Add a compact architecture diagram.
3. Add a regression command covering all climbing tests.
4. Ensure the existing tracking task documentation and behaviour are unchanged.
5. Produce a final validation matrix with pass/fail/evidence for every prior
   stage.

## Owner validation

A fresh user should be able to:

1. build/view a sampled ladder;
2. run the contact demo;
3. run the latch demo;
4. inspect relative-rung observations;
5. inspect reward transitions;
6. run the static hold;
7. run zero/random/scripted agents;
8. run a tiny PPO smoke test.

## Exit criteria

- Reproduction commands are complete.
- All design limitations are explicit.
- No knowledge required for validation exists only in the implementation
  agent's conversation history.

---

# Final validation matrix

The agent should maintain this table throughout implementation:

| Capability | Automated evidence | Manual evidence | Gate |
|---|---|---|---|
| Existing TeleopIt baseline | Existing tests/import | Existing task runs | Stage 0 |
| Pure-cylinder ladder | Geometry/sampler tests | Scene viewer | Stage 1 |
| Hand contact identity | Contact table tests | Contact viewer | Stage 2 |
| Attach/detach latch | State/constraint tests | Pull/release demo | Stage 3 |
| Relative rung observation | Transform/invariance tests | Debug lines/values | Stage 4 |
| Reward correctness | Table-driven oracle | Per-term scripted plot | Stage 5 |
| Brief static support | Hold metrics | Slow-motion viewer | Stage 6 |
| Reset/batch robustness | Repeated-reset tests | Zero/random/scripted play | Stage 7 |
| PPO plumbing | Finite smoke run/checkpoint | Playback | Stage 8 |
| Initial learnability | Baseline comparison | Multi-seed videos | Stage 9 |
| Reproducible handoff | Regression suite | Fresh-run checklist | Stage 10 |

# Explicit non-goals for the first implementation

- Production RGB-D or depth perception.
- Sim-to-real deployment.
- Realistic dexterous hands.
- Foot attachment constraints.
- Recompiling a different MJCF per episode.
- Broad domain randomization before fixed-task learning.
- Claiming physical G1 climbing feasibility from a MuJoCo result.
- Preserving the old 167D ONNX observation contract for the new climbing
  policy.

# References

- TeleopIt training architecture:
  https://botrunner64.github.io/Teleopit/tutorials/training/
- TeleopIt current G1 environment builder:
  https://raw.githubusercontent.com/BotRunner64/Teleopit/refs/heads/master/train_mimic/tasks/tracking/config/env.py
- TeleopIt base manager-based task configuration:
  https://raw.githubusercontent.com/BotRunner64/Teleopit/refs/heads/master/train_mimic/tasks/tracking/tracking_env_cfg.py
- mjlab dummy-agent and task examples:
  https://github.com/mujocolab/mjlab
- MuJoCo body/mocap and geometry reference:
  https://mujoco.readthedocs.io/en/stable/XMLreference.html
- MuJoCo runtime equality activation:
  https://mujoco.readthedocs.io/en/stable/programming/simulation.html

