#!/usr/bin/env python3
"""Play back a trained General-Climbing-G1 policy in simulation.

Usage:
    python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_100.pt

    python train_mimic/scripts/play_climb.py \
        --checkpoint logs/rsl_rl/g1_general_climbing/<run>/model_100.pt \
        --viewer viser
"""

from __future__ import annotations

import argparse
import os

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from train_mimic.app import (
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    resolve_device,
    validate_checkpoint_path,
)
from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
from train_mimic.tasks.climbing.config.smoke import make_climbing_smoke_env_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play trained G1 climbing policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--viewer",
        type=str,
        default="native",
        choices=["native", "viser"],
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-ladder",
        action="store_true",
        help="Use the Stage 8 pinned easy ladder layout (default for smoke checkpoints).",
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

    if args.smoke_ladder:
        env_cfg = make_climbing_smoke_env_cfg(
            num_envs=args.num_envs,
            seed=args.seed,
            play=True,
        )
    else:
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed

    device = resolve_device(args.device, torch)
    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if args.video:
        from mjlab.utils.wrappers import VideoRecorder

        log_dir = os.path.dirname(args.checkpoint)
        env = VideoRecorder(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=500,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    log_dir = os.path.dirname(args.checkpoint)
    agent_dict = build_runner_cfg_dict(agent_cfg, force_tensorboard=True)
    RunnerCls = runner_cls or MjlabOnPolicyRunner
    runner = RunnerCls(env, agent_dict, log_dir=log_dir, device=device)
    runner.load(args.checkpoint, map_location=device)
    policy = runner.get_inference_policy(device=device)

    if args.video:
        obs = env.get_observations()
        for _ in range(500):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)
    elif args.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        ViserPlayViewer(env, policy).run()

    env.close()


if __name__ == "__main__":
    main()
