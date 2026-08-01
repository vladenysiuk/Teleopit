#!/usr/bin/env python3
"""Debug entry point for General-Climbing-G1 staged development.

Stage 1 — ladder scene:

    mjpython train_mimic/scripts/debug_climb.py --mode scene --seed 42

Stage 2 — hand/rung contacts (G1 + ladder):

    mjpython train_mimic/scripts/debug_climb.py --mode contacts --seed 42

Stage 3 — attach/detach latch (G1 + ladder + connect constraints):

    mjpython train_mimic/scripts/debug_climb.py --mode latch --seed 42

Stage 4 — observation groups and relative-rung geometry:

    mjpython train_mimic/scripts/debug_climb.py --mode observations --seed 42

Stage 5 — reward terms, terminations, and metrics:

    mjpython train_mimic/scripts/debug_climb.py --mode rewards --seed 42

Stage 6 — static four-contact hold under gravity:

    mjpython train_mimic/scripts/debug_climb.py --mode hold-pose --seed 42

Stage 7 — end-to-end MDP agents and reset stress (no PPO):

    python train_mimic/scripts/debug_climb.py --mode zero --seed 42
    python train_mimic/scripts/debug_climb.py --mode random --seed 42 --num-envs 4
    python train_mimic/scripts/debug_climb.py --mode scripted-hold --seed 42
    python train_mimic/scripts/debug_climb.py --mode scripted-mdp --seed 42
    python train_mimic/scripts/debug_climb.py --mode reset-stress --seed 42 --num-resets 50

Stage 8 — tiny PPO smoke test (infrastructure validation, not learnability):

    python train_mimic/scripts/debug_climb.py --mode ppo-smoke --seed 42
    python train_mimic/scripts/train_climb.py --smoke
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


def _print_ladder_state(env, *, header: str) -> None:
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    runtime = LadderRuntime.get(env)
    if runtime.sample is None:
        print(f"{header}: ladder sample unavailable")
        return
    print(header)
    for env_idx in range(env.unwrapped.num_envs):
        for line in runtime.format_debug_lines(env_idx):
            print(line)
        active = runtime.sample.active_mask[env_idx].detach().cpu().tolist()
        heights = runtime.sample.rung_heights[env_idx].detach().cpu().tolist()
        print(f"  active_mask={active}")
        print(f"  rung_heights={heights}")


def _print_contact_state(env, *, header: str) -> None:
    from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState

    contacts = env.unwrapped.extras.get(ClimbContactState.EXTRA_KEY)
    if contacts is None:
        print(f"{header}: contact adapter unavailable")
        return
    contacts.update()
    print(header)
    for env_idx in range(env.unwrapped.num_envs):
        for line in contacts.format_debug_lines(env_idx):
            print(line)


def _print_latch_state(env, *, header: str) -> None:
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    latch = env.unwrapped.extras.get(ClimbLatchState.EXTRA_KEY)
    if latch is None:
        print(f"{header}: latch adapter unavailable")
        return
    print(header)
    for env_idx in range(env.unwrapped.num_envs):
        for line in latch.format_debug_lines(env_idx):
            print(line)


def run_scene_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.viewer.base import ViewerAction
    from mjlab.viewer.native.keys import KEY_N

    from train_mimic.tasks.climbing.config.ladder import LadderConfig
    from train_mimic.tasks.climbing.config.scene import make_ladder_scene_env_cfg
    from train_mimic.tasks.climbing.debug_vis import attach_ladder_debug_visualizer
    from train_mimic.tasks.climbing.ladder.state import LadderRuntime

    ladder_cfg = LadderConfig()
    env_cfg = make_ladder_scene_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        ladder_cfg=ladder_cfg,
        env_spacing=args.env_spacing,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    if args.export_scene is not None:
        export_path = Path(args.export_scene).expanduser().resolve()
        env.unwrapped.scene.write(export_path, zip=args.export_scene_zip)
        if args.export_scene_zip:
            print(f"Exported compiled scene zip: {export_path}.zip")
        else:
            print(f"Exported compiled scene: {export_path / 'scene.xml'}")

    if not args.no_debug_vis:
        attach_ladder_debug_visualizer(env.unwrapped)

    _print_ladder_state(env, header="[initial ladder sample]")

    class LadderSceneViewer(NativeMujocoViewer):
        def __init__(self, *viewer_args, **viewer_kwargs) -> None:
            super().__init__(*viewer_args, **viewer_kwargs)
            self._steps_since_reset = 0
            self._resample_requested = False

        def reset_environment(self) -> None:
            super().reset_environment()
            self._steps_since_reset = 0
            _print_ladder_state(self.env, header="[ladder reset]")

        def _handle_custom_action(self, action: ViewerAction, payload: object | None) -> bool:
            if action == ViewerAction.CUSTOM and payload == "RESAMPLE_LADDER":
                self._resample_requested = True
                return True
            return super()._handle_custom_action(action, payload)

        def _execute_step(self) -> bool:
            ok = super()._execute_step()
            if not ok:
                return False
            self._steps_since_reset += 1
            if self._resample_requested:
                self._resample_requested = False
                runtime = LadderRuntime.get(self.env.unwrapped)
                runtime.resample()
                _print_ladder_state(self.env, header="[ladder resample]")
            elif (
                args.reset_interval is not None
                and args.reset_interval > 0
                and self._steps_since_reset >= args.reset_interval
            ):
                self.request_reset()
            return True

    policy = lambda _obs: torch.zeros(env.unwrapped.num_envs, 0, device=env.unwrapped.device)

    viewer_holder: list[LadderSceneViewer] = []

    def key_callback(key: int) -> None:
        if key == KEY_N and viewer_holder:
            viewer_holder[0].request_action("CUSTOM", "RESAMPLE_LADDER")

    viewer = LadderSceneViewer(
        env,
        policy,
        key_callback=key_callback,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer_holder.append(viewer)
    viewer._show_debug_vis = not args.no_debug_vis
    print(
        "Controls: Enter=reset ladder, N=resample ladder, Space=pause, "
        "R=toggle debug overlay, Right=single step, Q=close window"
    )
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_contacts_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.viewer.native.keys import KEY_COMMA, KEY_DOWN, KEY_EQUAL, KEY_MINUS, KEY_PERIOD, KEY_UP

    from train_mimic.tasks.climbing.config.env import make_climbing_contacts_env_cfg
    from train_mimic.tasks.climbing.debug_probe import (
        place_probe_rung_near_hand,
        set_probe_rung_hand_only_collision,
    )
    from train_mimic.tasks.climbing.debug_vis import attach_contact_debug_visualizer

    env_cfg = make_climbing_contacts_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        env_spacing=args.env_spacing,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    probe_hand = "left"
    probe_rung_id = args.probe_rung
    separation_m = args.probe_separation

    # Probe cylinder collides with hand points only so it can follow a falling
    # robot without shoving through legs/torso. Robot dynamics stay live.
    set_probe_rung_hand_only_collision(env.unwrapped, probe_rung_id)

    if not args.no_debug_vis:
        attach_contact_debug_visualizer(env.unwrapped)

    place_probe_rung_near_hand(
        env.unwrapped,
        hand=probe_hand,
        rung_id=probe_rung_id,
        separation_m=separation_m,
    )
    _print_contact_state(env, header="[initial contact state]")

    class ContactDebugViewer(NativeMujocoViewer):
        def __init__(self, *viewer_args, **viewer_kwargs) -> None:
            super().__init__(*viewer_args, **viewer_kwargs)
            self.probe_hand = probe_hand
            self.probe_rung_id = probe_rung_id
            self.separation_m = separation_m

        def _execute_step(self) -> bool:
            place_probe_rung_near_hand(
                self.env.unwrapped,
                hand=self.probe_hand,
                rung_id=self.probe_rung_id,
                separation_m=self.separation_m,
            )
            ok = super()._execute_step()
            if ok:
                _print_contact_state(self.env, header="[contact step]")
            return ok

        def reset_environment(self) -> None:
            super().reset_environment()
            set_probe_rung_hand_only_collision(self.env.unwrapped, self.probe_rung_id)
            place_probe_rung_near_hand(
                self.env.unwrapped,
                hand=self.probe_hand,
                rung_id=self.probe_rung_id,
                separation_m=self.separation_m,
            )
            _print_contact_state(self.env, header="[contact reset]")

    policy = lambda _obs: torch.zeros(
        env.unwrapped.num_envs,
        env.action_manager.total_action_dim,
        device=env.unwrapped.device,
    )

    viewer_holder: list[ContactDebugViewer] = []

    def key_callback(key: int) -> None:
        if not viewer_holder:
            return
        viewer = viewer_holder[0]
        if key == KEY_UP:
            viewer.separation_m += 0.005
            print(f"probe separation -> {viewer.separation_m:.3f} m")
        elif key == KEY_DOWN:
            viewer.separation_m -= 0.005
            print(f"probe separation -> {viewer.separation_m:.3f} m")
        elif key == KEY_EQUAL:
            viewer.separation_m += 0.001
            print(f"probe separation -> {viewer.separation_m:.3f} m")
        elif key == KEY_MINUS:
            viewer.separation_m -= 0.001
            print(f"probe separation -> {viewer.separation_m:.3f} m")
        elif key == KEY_COMMA:
            viewer.probe_hand = "left"
            print("probe hand -> left")
        elif key == KEY_PERIOD:
            viewer.probe_hand = "right"
            print("probe hand -> right")

    viewer = ContactDebugViewer(
        env,
        policy,
        key_callback=key_callback,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer_holder.append(viewer)
    viewer._show_debug_vis = not args.no_debug_vis
    print(
        "Controls: Up/Down=±5mm probe separation, =/-=±1mm, ,/ .=left/right hand, "
        "Enter=reset, Space=pause, R=debug overlay, Right=single step, Q=close"
    )
    print(
        f"Probe rung=rung_{probe_rung_id:02d}; robot dynamics live; hand-only probe "
        f"collision; initial separation={separation_m:.3f} m along ladder +X from "
        f"{probe_hand} hand (laterally offset so the long rung misses the other hand)"
    )
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_zero_agent_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg
    from train_mimic.tasks.climbing.debug_mdp import (
        make_zero_action,
        print_rollout_summary,
        run_agent_rollout,
    )

    env_cfg = make_climbing_mdp_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        episode_length_s=args.episode_length,
        env_spacing=args.env_spacing,
        play=True,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    summary = run_agent_rollout(
        env,
        lambda _env, _step: make_zero_action(_env),
        num_steps=args.rollout_steps,
    )
    print_rollout_summary(summary, header="[zero agent]")
    env.close()


def run_random_agent_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg
    from train_mimic.tasks.climbing.debug_mdp import (
        make_random_action,
        print_rollout_summary,
        run_agent_rollout,
    )

    env_cfg = make_climbing_mdp_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        episode_length_s=args.episode_length,
        env_spacing=args.env_spacing,
        play=True,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed)
    summary = run_agent_rollout(
        env,
        lambda _env, _step: make_random_action(_env, generator=generator),
        num_steps=args.rollout_steps,
    )
    print_rollout_summary(summary, header="[random agent]")
    env.close()


def run_scripted_hold_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    from train_mimic.tasks.climbing.config.env import make_climbing_hold_pose_env_cfg
    from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
    from train_mimic.tasks.climbing.debug_mdp import (
        AssistedHoldScriptedAgent,
        print_rollout_summary,
        run_agent_rollout,
    )
    from train_mimic.tasks.climbing.debug_vis import attach_contact_debug_visualizer

    hold_cfg = HoldPoseConfig(hold_duration_s=max(args.hold_duration, 2.0))
    env_cfg = make_climbing_hold_pose_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        hold_cfg=hold_cfg,
        env_spacing=args.env_spacing,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    agent = AssistedHoldScriptedAgent(env.unwrapped, hold_cfg=hold_cfg)

    if not args.no_debug_vis:
        attach_contact_debug_visualizer(env.unwrapped)

    if args.headless:
        summary = run_agent_rollout(
            env.unwrapped,
            lambda e, s: agent(e, s),
            num_steps=args.rollout_steps,
        )
        print_rollout_summary(summary, header="[scripted-hold]")
        env.close()
        return

    class ScriptedHoldViewer(NativeMujocoViewer):
        def __init__(self, *viewer_args, **viewer_kwargs) -> None:
            super().__init__(*viewer_args, **viewer_kwargs)
            self._step_i = 0

        def _execute_step(self) -> bool:
            ok = super()._execute_step()
            if ok:
                self._step_i += 1
                if self._step_i % 10 == 0:
                    _print_latch_state(self.env, header="[scripted-hold step latch]")
            return ok

        def reset_environment(self) -> None:
            super().reset_environment()
            self._step_i = 0
            agent.reset()

    viewer = ScriptedHoldViewer(
        env,
        lambda _obs: agent(env.unwrapped, 0),
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer._show_debug_vis = not args.no_debug_vis
    print("Controls: Enter=reset+re-init assisted hold, Space=pause, Right=step, Q=close")
    print("Script phases:", " -> ".join(p.name for p in agent.script))
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_scripted_mdp_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg
    from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
    from train_mimic.tasks.climbing.debug_mdp import (
        FullMdpScriptedAgent,
        make_mdp_fault_injection_hook,
        pinned_ladder_cfg,
        print_rollout_summary,
        run_agent_rollout,
    )
    from train_mimic.tasks.climbing.debug_vis import attach_contact_debug_visualizer

    hold_cfg = HoldPoseConfig(hold_duration_s=max(args.hold_duration, 2.0))
    env_cfg = make_climbing_mdp_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        ladder_cfg=pinned_ladder_cfg(),
        episode_length_s=args.episode_length,
        env_spacing=args.env_spacing,
        play=True,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()
    agent = FullMdpScriptedAgent(env.unwrapped, hold_cfg=hold_cfg)
    fault_hook = make_mdp_fault_injection_hook(agent)

    if not args.no_debug_vis:
        attach_contact_debug_visualizer(env.unwrapped)

    if args.headless:
        summary = run_agent_rollout(
            env.unwrapped,
            lambda e, s: agent(e, s),
            num_steps=args.rollout_steps,
            pre_step_hook=fault_hook,
        )
        print_rollout_summary(summary, header="[scripted-mdp]")
        env.close()
        return

    class ScriptedMdpViewer(NativeMujocoViewer):
        def __init__(self, *viewer_args, **viewer_kwargs) -> None:
            super().__init__(*viewer_args, **viewer_kwargs)
            self._step_i = 0

        def _before_step(self) -> None:
            fault_hook(self.env.unwrapped, self._step_i)

        def _execute_step(self) -> bool:
            self._before_step()
            ok = super()._execute_step()
            if ok:
                self._step_i += 1
                if self._step_i % 10 == 0:
                    _print_latch_state(self.env, header="[scripted-mdp step latch]")
            return ok

        def reset_environment(self) -> None:
            super().reset_environment()
            self._step_i = 0
            agent.reset()

    viewer = ScriptedMdpViewer(
        env,
        lambda _obs: agent(env.unwrapped, 0),
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer._show_debug_vis = not args.no_debug_vis
    print("Controls: Enter=reset+re-init, Space=pause, Right=step, Q=close")
    print("Script phases:", " -> ".join(p.name for p in agent.script))
    print("Fall/invalid-attach root nudges are test-side fault injection, not agent actions.")
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_reset_stress_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg
    from train_mimic.tasks.climbing.debug_mdp import (
        make_random_action,
        pinned_ladder_cfg,
        print_reset_stress_stats,
        randomized_ladder_cfg,
        stress_random_resets,
    )
    from train_mimic.tasks.climbing.debug_vis import attach_contact_debug_visualizer

    ladder_cfg = pinned_ladder_cfg() if args.pinned_resets else randomized_ladder_cfg()
    env_cfg = make_climbing_mdp_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        ladder_cfg=ladder_cfg,
        episode_length_s=args.episode_length,
        env_spacing=args.env_spacing,
        play=True,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    if args.headless:
        stats = stress_random_resets(
            env.unwrapped,
            num_resets=args.num_resets,
            steps_after_reset=args.steps_after_reset,
            seed=args.seed,
            randomized_ladder=not args.pinned_resets,
        )
        print_reset_stress_stats(stats)
        env.close()
        return

    if not args.no_debug_vis:
        attach_contact_debug_visualizer(env.unwrapped)

    generator = torch.Generator(device=env.device)
    generator.manual_seed(args.seed)
    resets_done = 0

    def stress_policy(_obs):
        return make_random_action(env.unwrapped, generator=generator)

    class ResetStressViewer(NativeMujocoViewer):
        def reset_environment(self) -> None:
            nonlocal resets_done
            super().reset_environment()
            resets_done += 1
            _print_ladder_state(self.env, header=f"[reset stress #{resets_done}]")
            _print_latch_state(self.env, header="[reset stress latch]")

    viewer = ResetStressViewer(
        env,
        stress_policy,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer._show_debug_vis = not args.no_debug_vis
    print(
        "Controls: Enter=manual reset (watch ladder/latch/obs refresh), "
        "Space=pause, Right=step random action, Q=close"
    )
    print(
        f"Headless stress: python ... --mode reset-stress --headless "
        f"--num-resets {args.num_resets}"
    )
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_hold_pose_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    from train_mimic.tasks.climbing.config.env import make_climbing_hold_pose_env_cfg
    from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
    from train_mimic.tasks.climbing.debug_hold_pose import (
        HoldPoseMetrics,
        build_hold_action,
        classify_hold_failure,
        collect_hold_sample,
        initialize_hold_pose_scene,
        print_hold_summary,
        save_hold_pose_npz,
    )
    from train_mimic.tasks.climbing.debug_vis import attach_contact_debug_visualizer
    from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
    from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState

    hold_cfg = HoldPoseConfig(hold_duration_s=args.hold_duration)
    env_cfg = make_climbing_hold_pose_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        hold_cfg=hold_cfg,
        env_spacing=args.env_spacing,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    solution = initialize_hold_pose_scene(env.unwrapped, hold_cfg)
    metrics = HoldPoseMetrics(dt=float(env.unwrapped.step_dt))
    hold_steps = int(round(hold_cfg.hold_duration_s / env.unwrapped.step_dt))
    step_i = 0
    hold_complete = False

    if not args.no_debug_vis:
        attach_contact_debug_visualizer(env.unwrapped)

    _print_contact_state(env, header="[hold-pose initial contact]")
    _print_latch_state(env, header="[hold-pose initial latch]")
    print(
        f"[hold-pose IK] finite={solution.finite} "
        f"within_limits={solution.within_joint_limits} "
        f"hand_err={solution.hand_errors_m} foot_err={solution.foot_errors_m}"
    )

    pending_action = build_hold_action(env.unwrapped, solution)

    def hold_policy(_obs):
        return pending_action

    class HoldPoseDebugViewer(NativeMujocoViewer):
        def _execute_step(self) -> bool:
            nonlocal step_i, hold_complete, pending_action
            pending_action = build_hold_action(self.env.unwrapped, solution)
            ok = super()._execute_step()
            if not ok:
                return False
            step_i += 1
            if step_i >= hold_cfg.settle_steps:
                metrics.record(collect_hold_sample(self.env.unwrapped, hold_cfg))
            if step_i % 20 == 0 or step_i <= 3:
                contacts = ClimbContactState.get(self.env.unwrapped)
                contacts.update()
                latch = ClimbLatchState.get(self.env.unwrapped)
                print(
                    f"[hold step {step_i}] attached={latch.state.attached[0].tolist()} "
                    f"hand_rung={latch.state.rung_id[0].tolist()} "
                    f"pelvis_drop="
                    f"{(metrics.pelvis_drop_m[-1] if metrics.pelvis_drop_m else 0.0):.4f}"
                )
            if not hold_complete and step_i >= hold_steps:
                hold_complete = True
                summary = metrics.summary()
                failure = classify_hold_failure(
                    summary,
                    hold_cfg,
                    ik_ok=(
                        solution.finite
                        and solution.within_joint_limits
                        and float(solution.hand_errors_m.max()) < 0.06
                    ),
                )
                print_hold_summary(summary, failure_class=failure)
                if args.save_hold_npz is not None:
                    save_hold_pose_npz(
                        args.save_hold_npz,
                        solution,
                        metrics,
                        summary,
                    )
                    print(f"Saved hold-pose artifact: {args.save_hold_npz}")
                # Freeze at the end of the acceptance window so the viewer does
                # not keep sagging into a hang-from-hands / foot-wedge state.
                self.pause()
                print(
                    "[hold-pose] paused after hold window "
                    "(Space=resume, Enter=reset+re-solve IK, Q=close)"
                )
            return True

        def reset_environment(self) -> None:
            nonlocal step_i, hold_complete, solution, metrics, pending_action
            super().reset_environment()
            solution = initialize_hold_pose_scene(self.env.unwrapped, hold_cfg)
            metrics = HoldPoseMetrics(dt=float(self.env.unwrapped.step_dt))
            step_i = 0
            hold_complete = False
            pending_action = build_hold_action(self.env.unwrapped, solution)
            _print_latch_state(self.env, header="[hold-pose reset latch]")

    viewer = HoldPoseDebugViewer(
        env,
        hold_policy,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer._show_debug_vis = not args.no_debug_vis
    print(
        "Controls: Enter=reset+re-solve IK, Space=pause, Right=single step, Q=close"
    )
    print(
        "Hold uses one-shot pose write + force-seeded hand latches; "
        "no per-step root teleport (Stage 3 lesson). "
        "Auto-pauses after the hold window."
    )
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        if not hold_complete and metrics.pelvis_drop_m:
            summary = metrics.summary()
            failure = classify_hold_failure(
                summary,
                hold_cfg,
                ik_ok=solution.finite and solution.within_joint_limits,
            )
            print_hold_summary(summary, failure_class=failure, header="[hold-pose final]")
        env.close()


def run_rewards_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    from train_mimic.tasks.climbing.config.env import make_climbing_rewards_env_cfg
    from train_mimic.tasks.climbing.config.latch import LatchConfig
    from train_mimic.tasks.climbing.config.rewards import ClimbingRewardConfig
    from train_mimic.tasks.climbing.debug_rewards import (
        DEFAULT_REWARD_SCRIPT,
        RewardScriptPhase,
        apply_reward_script_phase,
        build_reward_action,
        initialize_rewards_debug_scene,
        print_episode_metrics,
        print_reward_breakdown,
    )

    latch_cfg = LatchConfig()
    reward_cfg = ClimbingRewardConfig()
    env_cfg = make_climbing_rewards_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        env_spacing=args.env_spacing,
        reward_cfg=reward_cfg,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    script = list(DEFAULT_REWARD_SCRIPT)
    phase_idx = 0
    phase_step = 0
    script_complete = False
    initialize_rewards_debug_scene(env.unwrapped)

    pending_action = torch.zeros(
        env.unwrapped.num_envs,
        env.action_manager.total_action_dim,
        device=env.unwrapped.device,
    )

    def reward_policy(_obs):
        return pending_action

    class RewardsDebugViewer(NativeMujocoViewer):
        def _current_phase(self) -> RewardScriptPhase:
            return script[phase_idx]

        def _prepare_action(self) -> None:
            phase = self._current_phase()
            apply_reward_script_phase(
                self.env.unwrapped,
                phase,
                phase_step=phase_step,
                latch_cfg=latch_cfg,
                script_complete=script_complete,
            )
            pending_action[:] = build_reward_action(
                self.env.unwrapped,
                latch_left=0.0 if script_complete else phase.latch_left,
                latch_right=0.0 if script_complete else phase.latch_right,
            )

        def _execute_step(self) -> bool:
            nonlocal phase_idx, phase_step, script_complete
            self._prepare_action()
            ok = super()._execute_step()
            if not ok:
                return False
            print_reward_breakdown(self.env, header=f"[reward step] phase={self._current_phase().name}")
            print_episode_metrics(self.env, header="[episode metrics]")
            if script_complete:
                return True
            phase_step += 1
            if phase_step >= self._current_phase().steps:
                phase_step = 0
                if phase_idx < len(script) - 1:
                    phase_idx += 1
                    print(f"[reward script] -> {self._current_phase().name}")
                else:
                    script_complete = True
                    print("[reward script] complete (holding still)")
            return True

        def reset_environment(self) -> None:
            nonlocal phase_idx, phase_step, script_complete
            super().reset_environment()
            phase_idx = 0
            phase_step = 0
            script_complete = False
            initialize_rewards_debug_scene(self.env.unwrapped)
            print_reward_breakdown(self.env, header="[reward reset]")

    viewer = RewardsDebugViewer(
        env,
        reward_policy,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    print("Controls: Enter=reset, Space=pause, Right=single step, Q=close")
    print("Script phases:", " -> ".join(p.name for p in script))
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_observations_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer

    from train_mimic.tasks.climbing.config.env import make_climbing_observations_env_cfg
    from train_mimic.tasks.climbing.debug_observations import (
        attach_observation_debug_visualizer,
        print_observation_shapes,
        print_relative_rung_values,
    )

    env_cfg = make_climbing_observations_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        env_spacing=args.env_spacing,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    if not args.no_debug_vis:
        attach_observation_debug_visualizer(env.unwrapped)

    print_observation_shapes(env.unwrapped, header="[initial observation shapes]")
    print_relative_rung_values(env.unwrapped, header="[initial relative rungs]")

    class ObservationsDebugViewer(NativeMujocoViewer):
        def _execute_step(self) -> bool:
            ok = super()._execute_step()
            if ok:
                print_observation_shapes(self.env, header="[observation step]")
                print_relative_rung_values(self.env, header="[relative rungs step]")
            return ok

        def reset_environment(self) -> None:
            super().reset_environment()
            print_observation_shapes(self.env, header="[observation reset]")
            print_relative_rung_values(self.env, header="[relative rungs reset]")

    policy = lambda _obs: torch.zeros(
        env.unwrapped.num_envs,
        env.action_manager.total_action_dim,
        device=env.unwrapped.device,
    )

    viewer = ObservationsDebugViewer(
        env,
        policy,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer._show_debug_vis = not args.no_debug_vis
    print(
        "Controls: Enter=reset, Space=pause, R=debug overlay, Right=single step, Q=close"
    )
    print("Move/rotate the robot root in the viewer to inspect relative-rung invariance.")
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_latch_mode(args: argparse.Namespace) -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer import NativeMujocoViewer
    from mjlab.viewer.native.keys import KEY_A, KEY_D, KEY_EQUAL, KEY_MINUS

    from train_mimic.tasks.climbing.config.env import make_climbing_latch_env_cfg
    from train_mimic.tasks.climbing.config.latch import LatchConfig
    from train_mimic.tasks.climbing.debug_latch import (
        DEFAULT_LATCH_SCRIPT,
        LatchScriptPhase,
        apply_latch_script_phase,
        build_latch_action,
        initialize_latch_debug_scene,
    )
    from train_mimic.tasks.climbing.debug_vis import attach_contact_debug_visualizer

    latch_cfg = LatchConfig()
    env_cfg = make_climbing_latch_env_cfg(
        num_envs=args.num_envs,
        seed=args.seed,
        env_spacing=args.env_spacing,
        latch_cfg=latch_cfg,
    )
    device = args.device or "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    script = list(DEFAULT_LATCH_SCRIPT)
    phase_idx = 0
    phase_step = 0
    manual_left = 0.0
    manual_right = 0.0
    auto_run = not args.manual_latch

    initialize_latch_debug_scene(env.unwrapped)

    if not args.no_debug_vis:
        attach_contact_debug_visualizer(env.unwrapped)

    _print_contact_state(env, header="[initial contact state]")
    _print_latch_state(env, header="[initial latch state]")

    viewer_holder: list[LatchDebugViewer] = []
    pending_action = torch.zeros(
        env.unwrapped.num_envs,
        env.action_manager.total_action_dim,
        device=env.unwrapped.device,
    )

    def latch_policy(_obs):
        return pending_action

    class LatchDebugViewer(NativeMujocoViewer):
        def __init__(self, *viewer_args, **viewer_kwargs) -> None:
            super().__init__(*viewer_args, **viewer_kwargs)
            self.auto_run = auto_run

        def _current_phase(self) -> LatchScriptPhase:
            return script[phase_idx]

        def _prepare_action(self) -> None:
            phase = self._current_phase()

            if self.auto_run:
                apply_latch_script_phase(
                    self.env.unwrapped,
                    phase,
                    phase_step=phase_step,
                )
                pending_action[:] = build_latch_action(
                    self.env.unwrapped,
                    latch_left=phase.latch_left,
                    latch_right=phase.latch_right,
                )
            else:
                apply_latch_script_phase(
                    self.env.unwrapped,
                    LatchScriptPhase(
                        name=phase.name,
                        hand=phase.hand,
                        rung_id=phase.rung_id,
                        separation_m=phase.separation_m,
                        latch_left=manual_left,
                        latch_right=manual_right,
                        steps=1,
                        motion_mode=phase.motion_mode,
                        approach_step_m=phase.approach_step_m,
                        pull_step_m=phase.pull_step_m,
                    ),
                    phase_step=phase_step,
                )
                pending_action[:] = build_latch_action(
                    self.env.unwrapped,
                    latch_left=manual_left,
                    latch_right=manual_right,
                )

        def _execute_step(self) -> bool:
            nonlocal phase_idx, phase_step
            self._prepare_action()
            ok = super()._execute_step()
            if not ok:
                return False
            if self.auto_run:
                phase_step += 1
                if phase_step >= self._current_phase().steps:
                    phase_step = 0
                    phase_idx = min(phase_idx + 1, len(script) - 1)
                    print(f"[latch script] -> {self._current_phase().name}")
            _print_contact_state(self.env, header="[latch step contact]")
            _print_latch_state(self.env, header="[latch step state]")
            return True

        def reset_environment(self) -> None:
            nonlocal phase_idx, phase_step
            super().reset_environment()
            phase_idx = 0
            phase_step = 0
            initialize_latch_debug_scene(self.env.unwrapped)
            _print_contact_state(self.env, header="[latch reset contact]")
            _print_latch_state(self.env, header="[latch reset state]")

    def key_callback(key: int) -> None:
        nonlocal manual_left, manual_right, phase_idx, phase_step
        if not viewer_holder:
            return
        viewer = viewer_holder[0]
        if key == KEY_A:
            manual_left = latch_cfg.attach_threshold + 0.2
            print(f"manual latch left attach -> {manual_left:.2f}")
        elif key == KEY_D:
            manual_left = latch_cfg.detach_threshold - 0.2
            print(f"manual latch left detach -> {manual_left:.2f}")
        elif key == KEY_EQUAL:
            manual_left = 0.0
            manual_right = 0.0
            print("manual latch neutral")
        elif key == KEY_MINUS:
            viewer.auto_run = not viewer.auto_run
            print(f"auto script playback -> {viewer.auto_run}")

    viewer = LatchDebugViewer(
        env,
        latch_policy,
        key_callback=key_callback,
        verbosity=viewer_kwargs_verbosity(args.verbose),
    )
    viewer_holder.append(viewer)
    viewer._show_debug_vis = not args.no_debug_vis
    print(
        "Controls: auto script by default; `-` toggles auto/manual, "
        "`A`=attach left, `D`=detach left, `=`=neutral, Enter=reset, "
        "Space=pause, Right=single step, Q=close"
    )
    print("Script phases:", " -> ".join(p.name for p in script))
    if sys.platform == "darwin":
        print("macOS: use mjpython (not python) for the native MuJoCo window.")
    try:
        viewer.run()
    finally:
        env.close()


def run_ppo_smoke_mode(args: argparse.Namespace) -> None:
    from train_mimic.tasks.climbing.config.smoke import (
        SMOKE_MAX_ITERATIONS,
        SMOKE_NUM_ENVS,
        SMOKE_SAVE_INTERVAL,
    )
    from train_mimic.tasks.climbing.ppo_smoke import run_ppo_smoke

    num_envs = args.num_envs if args.num_envs != 1 else SMOKE_NUM_ENVS
    max_iterations = args.rollout_steps if args.rollout_steps != 120 else SMOKE_MAX_ITERATIONS
    report = run_ppo_smoke(
        num_envs=num_envs,
        max_iterations=max_iterations,
        save_interval=SMOKE_SAVE_INTERVAL,
        seed=args.seed,
        device=args.device,
    )
    print("[ppo-smoke] device:", report.device)
    print("[ppo-smoke] log_dir:", report.log_dir)
    print("[ppo-smoke] checkpoint:", report.checkpoint_path)
    print("[ppo-smoke] completed_episodes:", report.completed_episodes)
    print("[ppo-smoke] reward_variance:", f"{report.reward_variance:.6f}")
    print("[ppo-smoke] latch_action_std:", f"{report.latch_action_std:.4f}")
    print("[ppo-smoke] playback_ok:", report.playback_ok)
    if report.mean_rewards:
        print("[ppo-smoke] mean_reward:", f"{statistics.mean(report.mean_rewards):.4f}")
    if report.extra.get("term_means"):
        terms = report.extra["term_means"]
        joined = ", ".join(f"{k}={v:.4f}" for k, v in sorted(terms.items()))
        print("[ppo-smoke] reward_terms:", joined)
    if report.reward_scale.flagged_terms:
        print("[ppo-smoke] reward_scale_flags:", ", ".join(report.reward_scale.flagged_terms))
        for note in report.reward_scale.notes:
            print("[ppo-smoke] note:", note)
    if not report.ok:
        raise SystemExit("[ppo-smoke] smoke report failed exit criteria")


def viewer_kwargs_verbosity(verbose: bool) -> int:
    from mjlab.viewer.base import VerbosityLevel

    return VerbosityLevel.INFO if verbose else VerbosityLevel.SILENT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug General-Climbing-G1 development stages.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "scene",
            "contacts",
            "latch",
            "observations",
            "rewards",
            "hold-pose",
            "zero",
            "random",
            "scripted-hold",
            "scripted-mdp",
            "scripted",
            "reset-stress",
            "ppo-smoke",
        ],
        help="Debug mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for ladder sampling")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments")
    parser.add_argument("--env-spacing", type=float, default=4.0, help="Spacing between env copies")
    parser.add_argument(
        "--reset-interval",
        type=int,
        default=None,
        help="Optional automatic full reset interval in viewer steps (scene mode)",
    )
    parser.add_argument("--device", type=str, default=None, help="Simulation device (default: cpu)")
    parser.add_argument(
        "--no-debug-vis",
        action="store_true",
        help="Disable debug overlay spheres",
    )
    parser.add_argument(
        "--export-scene",
        type=str,
        default=None,
        help="Write compiled MJCF to this directory before opening the viewer (scene mode)",
    )
    parser.add_argument(
        "--export-scene-zip",
        action="store_true",
        help="With --export-scene, write <path>.zip instead of a directory",
    )
    parser.add_argument(
        "--probe-rung",
        type=int,
        default=3,
        help="Active rung index moved near the hand in contacts mode",
    )
    parser.add_argument(
        "--probe-separation",
        type=float,
        default=0.08,
        help="Initial probe rung separation from hand point along ladder +X (meters)",
    )
    parser.add_argument(
        "--manual-latch",
        action="store_true",
        help="Disable auto latch script and use A/D/= manual latch keys (latch mode)",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=2.5,
        help="Gravity hold duration in seconds (hold-pose mode, >=2)",
    )
    parser.add_argument(
        "--save-hold-npz",
        type=str,
        default=None,
        help="Optional path to save hold-pose IK + metrics NPZ (hold-pose mode)",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=200,
        help="Headless rollout length for zero/random/scripted modes",
    )
    parser.add_argument(
        "--episode-length",
        type=float,
        default=5.0,
        help="Episode length in seconds for Stage 7 MDP modes",
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        default=200,
        help="Number of randomized resets in reset-stress headless mode",
    )
    parser.add_argument(
        "--steps-after-reset",
        type=int,
        default=3,
        help="Zero-action steps after each reset in reset-stress headless mode",
    )
    parser.add_argument(
        "--pinned-resets",
        action="store_true",
        help="Use pinned ladder geometry for reset-stress (reproducibility only)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run zero/random/scripted/reset-stress without opening a viewer",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose viewer logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "scene":
        run_scene_mode(args)
        return
    if args.mode == "contacts":
        run_contacts_mode(args)
        return
    if args.mode == "latch":
        run_latch_mode(args)
        return
    if args.mode == "observations":
        run_observations_mode(args)
        return
    if args.mode == "rewards":
        run_rewards_mode(args)
        return
    if args.mode == "hold-pose":
        run_hold_pose_mode(args)
        return
    if args.mode == "zero":
        run_zero_agent_mode(args)
        return
    if args.mode == "random":
        run_random_agent_mode(args)
        return
    if args.mode == "scripted-hold":
        run_scripted_hold_mode(args)
        return
    if args.mode == "scripted-mdp" or args.mode == "scripted":
        run_scripted_mdp_mode(args)
        return
    if args.mode == "reset-stress":
        run_reset_stress_mode(args)
        return
    if args.mode == "ppo-smoke":
        run_ppo_smoke_mode(args)
        return
    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main(sys.argv[1:])
