#!/usr/bin/env python3
"""Play back a trained General-Climbing-G1 policy in simulation.

Usage:
    # Stage 9 easy checkpoint (assisted hold reset + fixed ladder — matches train --easy)
    python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
        --easy

    # Record headless video (full easy episode by default, saved next to checkpoint)
    MUJOCO_GL=egl python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
        --easy --video --device cuda:0

    # Stage 8 smoke checkpoint (pinned ladder, no curriculum reset)
    python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_100.pt \
        --smoke-ladder

    # Browser viewer over SSH
    python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
        --easy --viewer viser
"""

from __future__ import annotations

import argparse
import os

from train_mimic.warp_patches import apply_mujoco_warp_sensor_patches

apply_mujoco_warp_sensor_patches()

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from train_mimic.app import (
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    resolve_device,
    validate_checkpoint_path,
)
from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
from train_mimic.tasks.climbing.config.easy import make_climbing_easy_env_cfg
from train_mimic.tasks.climbing.config.smoke import make_climbing_smoke_env_cfg


def resolve_play_video_length(env_cfg, override: int | None) -> int:
    """Policy steps to record; defaults to one full episode."""
    if override is not None:
        return max(1, override)
    step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    return max(1, int(round(env_cfg.episode_length_s / step_dt)))


def configure_headless_video_rendering() -> None:
    """Default EGL for headless rgb_array capture on servers without a display."""
    if "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
        print("[INFO] --video enabled, MUJOCO_GL not set. Defaulting to MUJOCO_GL=egl.")
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        print(
            "[INFO] --video enabled, PYOPENGL_PLATFORM not set. "
            "Defaulting to PYOPENGL_PLATFORM=egl."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play trained G1 climbing policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--viewer",
        type=str,
        default="native",
        choices=["native", "viser"],
        help="Ignored when --video is set (headless recording).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Record rgb_array video instead of opening an interactive viewer.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=None,
        help="Policy steps to record (default: one full episode from env cfg).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default=None,
        help="Output directory for mp4 (default: <checkpoint_dir>/videos/play).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    env_preset = parser.add_mutually_exclusive_group()
    env_preset.add_argument(
        "--easy",
        action="store_true",
        help=(
            "Use the Stage 9 easy MDP (fixed ladder, assisted hold reset, hand attach). "
            "Match train_climb.py --easy checkpoints."
        ),
    )
    env_preset.add_argument(
        "--smoke-ladder",
        action="store_true",
        help="Use the Stage 8 smoke ladder (pinned layout, no curriculum reset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        _load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()

    try:
        validate_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

    configure_torch_backends()

    _task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
        CLIMBING_TASK_ID,
        play=True,
        load_env_cfg=_load_env_cfg,
        load_rl_cfg=_load_rl_cfg,
        load_runner_cls=_load_runner_cls,
    )

    if args.easy:
        env_cfg = make_climbing_easy_env_cfg(
            num_envs=args.num_envs,
            seed=args.seed,
            play=True,
        )
    elif args.smoke_ladder:
        env_cfg = make_climbing_smoke_env_cfg(
            num_envs=args.num_envs,
            seed=args.seed,
            play=True,
        )
    else:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed

    if args.video and args.num_envs != 1:
        print("Error: --video requires --num_envs 1")
        raise SystemExit(1)

    video_length = resolve_play_video_length(env_cfg, args.video_length) if args.video else 0
    log_dir = os.path.dirname(args.checkpoint)
    video_folder = args.video_folder or os.path.join(log_dir, "videos", "play")

    if args.video:
        configure_headless_video_rendering()
        os.makedirs(video_folder, exist_ok=True)
        step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
        print(
            f"[INFO] Recording {video_length} policy steps "
            f"({video_length * step_dt:.1f}s) to {video_folder}"
        )

    device = resolve_device(args.device, torch)
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.video:
        from mjlab.utils.wrappers import VideoRecorder

        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_dict = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
    RunnerCls = runner_cls or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    if args.video:
        obs = env.get_observations()
        for _ in range(video_length):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        print(f"[INFO] Video saved under {video_folder}")
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
