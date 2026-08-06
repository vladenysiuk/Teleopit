<p align="center">
  <img src="assets/teleopit_logo.jpg" width="80" alt="Teleopit">
</p>

<h1 align="center">Teleopit — Climbing</h1>

<p align="center">
  A fork of <a href="https://github.com/BotRunner64/Teleopit">Teleopit</a> for learning Unitree G1 ladder-climbing tasks in MuJoCo.
  <br/>
  Trains <code>General-Climbing-G1</code> with mjlab RL (latches, contacts, curriculum) on top of the Teleopit / train_mimic stack.
</p>

<p align="center">
  <a href="docs/docs/tutorials/climbing.md">Climbing Tutorial</a> &bull;
  <a href="docs/docs/reference/climbing-rl.md">Climbing RL Reference</a> &bull;
  <a href="docs/docs/reference/climbing-simulator.md">Climbing Simulator</a> &bull;
  <a href="https://BotRunner64.github.io/Teleopit/">Upstream Docs</a>
</p>

---

## Quick Start — Ladder Climbing

**Docs:** [Climbing Tutorial](docs/docs/tutorials/climbing.md) · [Climbing RL Reference](docs/docs/reference/climbing-rl.md) · [Climbing Simulator](docs/docs/reference/climbing-simulator.md)

`General-Climbing-G1` is a separate mjlab RL task. It does **not** share the teleoperation 167D tracking ONNX contract.

**1. Install**

```bash
pip install -e ".[train]"
```

**2. Download robot assets**

```bash
python scripts/setup/download_assets.py --only robots
```

Canonical model: `assets/robots/unitree_g1/g1_29dof.xml`.

**3. Validate, then train**

```bash
# Automated regression
python scripts/dev/run_climbing_regression.py

# Interactive scene check (macOS: mjpython)
mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42

# Training
python train_mimic/scripts/train_climb.py --easy
python train_mimic/scripts/train_climb.py --medium   # easy MDP, full 12-rung ladder
```

See the [climbing tutorial](docs/docs/tutorials/climbing.md) for debug modes, presets, and playback. For rewards, PPO, curriculum, and model layout, see the [climbing RL reference](docs/docs/reference/climbing-rl.md).

Online mirror: **[BotRunner64.github.io/Teleopit/tutorials/climbing](https://BotRunner64.github.io/Teleopit/tutorials/climbing)**

---

## Upstream Teleopit (Tracking / Teleoperation)

This fork retains the original whole-body teleoperation stack (BVH / Pico 4 → GMR retargeting → ONNX policy → MuJoCo or Unitree G1). Use it when you need sim2sim tracking or sim2real teleop, not climbing RL.

### Minimal Sim2Sim

```bash
pip install -e .
pip install modelscope
python scripts/setup/download_assets.py --only robots gmr ckpt bvh

python scripts/run/run_sim.py \
    controller.policy_path=track.onnx \
    input.bvh_file=data/sample_bvh/aiming1_subject1.bvh
```

Add `'viewers=[sim2sim,camera]'` for the D435i RGB camera view. Sim2real defaults to `viewers=none`; use `viewers=retarget` for an optional retarget window.

### Pico Motion Recording

```bash
pip install -e '.[pico4]'
python scripts/run/record_pico_motion.py
```

Terminal: `R` start, `S` save, `D` discard, `N` new name, `Q` quit. Clips land in `data/pico_motion/clips/`.

```bash
python train_mimic/scripts/data/build_dataset.py \
    --spec data/pico_motion/pico_recorded.yaml --force
```

### Sim2Real HDF5 Recording

```bash
pip install -e '.[recording]'
python scripts/run/run_sim2real.py --config-name sim2real_record \
    controller.policy_path=track.onnx \
    recording.task="walk forward"
```

Episodes under `data/recordings/sim2real_hdf5/`; MP4 sidecars under `videos/`.

Upstream docs: **[BotRunner64.github.io/Teleopit](https://BotRunner64.github.io/Teleopit/)** (Pico Sim2Sim / Sim2Real, tracking training, architecture).

## Changelog

### v0.4.0 (2026-06-25)

- Improved Pico realtime control with pico-bridge 0.2.1, `ARMS` mode, armed sim2real mocap entry, and retargeter-preserving pause/arms resets.
- Added optional LinkerHand L6/O6 sim2real control, including Pico gripper input and low-latency L6 `vr_hand_pose`.
- Added manual Pico sim2real HDF5 recording and an interactive Pico motion recorder for training NPZ clips.
- Refined the training data path with minimal HDF5 shards, explicit precompute, rewind sampling, and updated tracking rewards.

### v0.3.0 (2026-05-12)

- Consolidated realtime input around pico-bridge 0.2.0 and removed the old ZMQ/onboard Pico path.
- Unified sim/sim2real reference buffering, resume realignment, and velocity smoothing.
- Added UDP BVH realtime input, online sim config, multi-viewer support, and fixed camera viewing.
- Split sim2real reference/safety runtime modules and updated the G1 MuJoCo camera asset.

### v0.2.0 (2026-04-03)

- Added Pico 4 teleoperation through pico-bridge and the G1 Bridge SDK.
- Added offline playback keyboard controls, Pico sim2sim mode control, and a standalone standing controller.
- Improved realtime mocap buffering/catch-up and upgraded the released model to the 30k checkpoint.

### v0.1.1 (2026-03-28)

- Dataset shard-only refactor
- External asset management (ModelScope), repository slimming

### v0.1.0 (2026-03-25)

- Initial public release: General-Tracking-G1 training, ONNX sim2sim inference, Pico 4 VR teleoperation, Unitree G1 hardware deployment

## License

[Apache 2.0](LICENSE)
