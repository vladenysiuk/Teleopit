#!/usr/bin/env python3
"""Play back a trained General-Climbing-G1 policy in simulation.

Usage:
    # Stage 9 easy checkpoint (assisted hold reset + fixed ladder — matches train --easy)
    python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
        --easy

    # Record headless multi-clip video (reseeds each clip; one Full HD mp4 at 30% speed)
    MUJOCO_GL=egl python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_800.pt \
        --easy --video --seed 42

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
from pathlib import Path

import numpy as np

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

# Default multi-seed validation capture: several reset samples in one mp4.
DEFAULT_VIDEO_CLIPS = 8
DEFAULT_VIDEO_SPEED = 0.3
DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080


def resolve_play_video_length(env_cfg, override: int | None) -> int:
    """Policy steps to record per clip; defaults to one full episode."""
    if override is not None:
        return max(1, override)
    step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    return max(1, int(round(env_cfg.episode_length_s / step_dt)))


def resolve_video_fps(step_dt: float, speed: float) -> float:
    """Output fps for wall-clock playback at the requested fraction of realtime."""
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    if speed <= 0.0:
        raise ValueError(f"video speed must be positive, got {speed}")
    realtime_fps = 1.0 / step_dt
    return max(1.0, realtime_fps * speed)


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


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Convert an rgb_array frame to HxWx3 uint8."""
    rgb = frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb


def _set_render_env_idx(unwrapped, env_idx: int) -> None:
    """Point the offscreen camera at a parallel env and clear neighbor cache."""
    unwrapped.cfg.viewer.env_idx = int(env_idx)
    renderer = getattr(unwrapped, "_offline_renderer", None)
    if renderer is not None:
        renderer._extra_env_ids = None


def record_play_video(
    *,
    env,
    policy,
    torch,
    video_path: Path,
    video_length: int,
    num_clips: int,
    video_speed: float,
    seed: int,
) -> Path:
    """Roll out ``num_clips`` reseeds and write one concatenated mp4 at ``video_speed``."""
    import mediapy as media
    from tensordict import TensorDict

    unwrapped = env.unwrapped
    step_dt = float(unwrapped.step_dt)
    fps = resolve_video_fps(step_dt, video_speed)
    frames: list[np.ndarray] = []

    # Keep validation clips focused on the tracked robot.
    unwrapped.cfg.viewer.max_extra_envs = 0

    for clip_idx in range(num_clips):
        clip_seed = seed + clip_idx
        render_env_idx = clip_idx % unwrapped.num_envs
        _set_render_env_idx(unwrapped, render_env_idx)

        obs_dict, _ = unwrapped.reset(seed=clip_seed)
        obs = TensorDict(obs_dict, batch_size=[unwrapped.num_envs])

        print(
            f"[INFO] Video clip {clip_idx + 1}/{num_clips}: "
            f"seed={clip_seed}, env_idx={render_env_idx}, steps={video_length}"
        )
        for _ in range(video_length):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            frame = unwrapped.render()
            if frame is not None:
                frames.append(_normalize_frame(frame))

    if not frames:
        raise RuntimeError("No frames were captured; check render_mode='rgb_array'.")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(video_path), frames, fps=fps)
    wall_s = len(frames) / fps
    sim_s = len(frames) * step_dt
    print(
        f"[INFO] Saved {len(frames)} frames ({num_clips} clips, "
        f"{sim_s:.1f}s sim → {wall_s:.1f}s @ {video_speed:.0%} speed, "
        f"{fps:.1f} fps) to {video_path}"
    )
    return video_path


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
        help=(
            "Record a single mp4 of multiple reseeds (default: "
            f"{DEFAULT_VIDEO_CLIPS} clips at {DEFAULT_VIDEO_SPEED:.0%} speed)."
        ),
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=None,
        help="Policy steps per clip (default: one full episode from env cfg).",
    )
    parser.add_argument(
        "--video-clips",
        type=int,
        default=DEFAULT_VIDEO_CLIPS,
        help=f"Number of reset samples to concatenate (default: {DEFAULT_VIDEO_CLIPS}).",
    )
    parser.add_argument(
        "--video-speed",
        type=float,
        default=DEFAULT_VIDEO_SPEED,
        help=(
            "Playback speed as a fraction of realtime "
            f"(default: {DEFAULT_VIDEO_SPEED} = {DEFAULT_VIDEO_SPEED:.0%})."
        ),
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default=None,
        help="Output directory for mp4 (default: <checkpoint_dir>/videos/play).",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help=f"Offscreen render width in pixels (default: {DEFAULT_VIDEO_WIDTH}).",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help=f"Offscreen render height in pixels (default: {DEFAULT_VIDEO_HEIGHT}).",
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

    if args.video:
        if args.video_clips < 1:
            print("Error: --video-clips must be >= 1")
            raise SystemExit(1)
        if args.video_speed <= 0.0:
            print("Error: --video-speed must be > 0")
            raise SystemExit(1)
        if args.video_width < 1 or args.video_height < 1:
            print("Error: --video-width and --video-height must be >= 1")
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

    video_length = resolve_play_video_length(env_cfg, args.video_length) if args.video else 0
    log_dir = os.path.dirname(args.checkpoint)
    video_folder = Path(args.video_folder or os.path.join(log_dir, "videos", "play"))
    video_path = video_folder / "play_climb.mp4"

    if args.video:
        configure_headless_video_rendering()
        env_cfg.viewer.width = args.video_width
        env_cfg.viewer.height = args.video_height
        video_folder.mkdir(parents=True, exist_ok=True)
        step_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
        fps = resolve_video_fps(step_dt, args.video_speed)
        print(
            f"[INFO] Recording {args.video_clips} clips × {video_length} policy steps "
            f"({video_length * step_dt:.1f}s sim/clip) at {args.video_speed:.0%} speed "
            f"({fps:.1f} fps, {args.video_width}x{args.video_height}) → {video_path}"
        )

    device = resolve_device(args.device, torch)
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_dict = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
    RunnerCls = runner_cls or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    if args.video:
        record_play_video(
            env=env,
            policy=policy,
            torch=torch,
            video_path=video_path,
            video_length=video_length,
            num_clips=args.video_clips,
            video_speed=args.video_speed,
            seed=args.seed,
        )
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
