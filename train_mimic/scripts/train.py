#!/usr/bin/env python3
"""Train G1 whole-body tracking policy with mjlab + rsl_rl PPO.

Usage:
    python train_mimic/scripts/train.py \
        --num_envs 4096 --max_iterations 18000 \
        --motion_file data/datasets_precomputed

    # Quick verification
    python train_mimic/scripts/train.py \
        --num_envs 64 --max_iterations 100 \
        --motion_file data/datasets_precomputed

    # With W&B logging
    python train_mimic/scripts/train.py \
        --num_envs 4096 --max_iterations 30000 \
        --motion_file data/datasets_precomputed \
        --logger wandb

    # With SwanLab logging
    python train_mimic/scripts/train.py \
        --num_envs 4096 --max_iterations 30000 \
        --motion_file data/datasets_precomputed \
        --logger swanlab

    # Resume for additional iterations
    python train_mimic/scripts/train.py \
        --resume logs/rsl_rl/g1_general_tracking/<run>/model_12000.pt \
        --max_iterations 18000 \
        --motion_file data/datasets_precomputed
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
from datetime import datetime
from typing import Any, Sequence

from train_mimic.app import (
    DEFAULT_TASK,
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    validate_motion_file,
)
from train_mimic.distributed_launch import (
    add_multi_gpu_arguments,
    build_launcher_env as _build_launcher_env,
    build_torchrun_command as _build_torchrun_command,
    destroy_process_group as _destroy_process_group,
    filtered_argv_for_worker as _filtered_argv_for_worker,
    is_distributed_env as _is_distributed_env,
    is_main_process as _is_main_process,
    launch_multi_gpu as _launch_multi_gpu,
    normalize_multi_gpu_args,
    resolve_distributed_device as _resolve_device,
    resolve_worker_seed as _resolve_worker_seed,
    should_launch_multi_gpu as _should_launch_multi_gpu,
    terminate_worker_group as _terminate_worker_group,
    validate_multi_gpu_args as _validate_multi_gpu_args,
    wait_process as _wait_process,
)
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.config.env import (
    make_g1_training_robot_cfg,
    resolve_g1_training_xml,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train G1 tracking policy (mjlab).")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help=(
            "Number of learning iterations to run in this invocation. When resuming from model_N.pt, "
            "this adds more iterations on top of N."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "swanlab"],
        help="Experiment logger backend (default: tensorboard)",
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--motion_file", type=str, default=None,
                        help="Precomputed training dataset root containing Teleopit shard_*.h5 files, searched recursively")
    parser.add_argument(
        "--robot_xml",
        type=str,
        default=None,
        help=(
            "MuJoCo XML used for the G1 training robot "
            "(default: assets/robots/unitree_g1/g1_29dof.xml)"
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to checkpoint to resume from. With --resume, --max_iterations means additional "
            "iterations to run after loading the checkpoint."
        ),
    )
    parser.add_argument("--sampling_mode", type=str, default=None,
                        choices=["uniform", "start", "rewind"],
                        help="Motion sampling mode (default: from task config)")
    parser.add_argument("--rewind_prob", type=float, default=None,
                        help="Rewind sampling probability for failed episodes")
    parser.add_argument("--rewind_min_steps", type=int, default=None,
                        help="Minimum policy steps to rewind for rewind sampling")
    parser.add_argument("--rewind_max_steps", type=int, default=None,
                        help="Maximum policy steps to rewind for rewind sampling")
    parser.add_argument("--device", type=str, default=None)
    add_multi_gpu_arguments(parser)
    parser.add_argument("--video", action="store_true",
                        help="Record periodic videos during training")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="Task id to train (default: %(default)s)")
    parser.add_argument("--video_interval", type=int, default=2000,
                        help="Record a video every N iterations (default: 2000)")
    parser.add_argument("--video_length", type=int, default=200,
                        help="Number of steps per video clip (default: 200)")
    return parser.parse_args(argv)


def _configure_experiment_logger(
    *,
    logger_name: str,
    agent_cfg: Any,
    env_cfg: Any,
    log_dir: str,
) -> bool:
    """Configure the training logger and return whether SwanLab was started."""
    if logger_name == "tensorboard":
        agent_cfg.logger = "tensorboard"
        return False

    if logger_name == "wandb":
        agent_cfg.logger = "wandb"
        agent_cfg.wandb_project = agent_cfg.experiment_name
        return False

    if logger_name != "swanlab":
        raise ValueError(f"Unsupported logger '{logger_name}'")

    agent_cfg.logger = "tensorboard"
    if not _is_main_process():
        return False

    try:
        import swanlab
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "swanlab package is required for --logger swanlab. Install it with `pip install swanlab`."
        ) from None

    swanlab.init(
        project=agent_cfg.experiment_name,
        name=os.path.basename(log_dir),
        log_dir=log_dir,
        config={
            "experiment_name": agent_cfg.experiment_name,
            "motion_file": env_cfg.commands["motion"].motion_file,
            "robot_xml": getattr(env_cfg, "robot_xml", None),
            "num_envs": env_cfg.scene.num_envs,
            "max_iterations": agent_cfg.max_iterations,
            "sampling_mode": env_cfg.commands["motion"].sampling_mode,
            "rewind_prob": env_cfg.commands["motion"].rewind_prob,
            "rewind_min_steps": env_cfg.commands["motion"].rewind_min_steps,
            "rewind_max_steps": env_cfg.commands["motion"].rewind_max_steps,
        },
    )
    swanlab.sync_tensorboard_torch(types=["scalar", "scalars", "image", "text"])
    return True


def _run_worker(args: argparse.Namespace) -> None:
    (
        torch,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    env: Any | None = None
    rank = os.environ.get("RANK", "0")

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        print(f"[INFO] Rank {rank} received signal {signum}, shutting down...")
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        _destroy_process_group(torch)
        raise KeyboardInterrupt

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    configure_torch_backends()

    # Load configs from registry
    _task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
        args.task,
        load_env_cfg=load_env_cfg,
        load_rl_cfg=load_rl_cfg,
        load_runner_cls=load_runner_cls,
    )

    # CLI overrides
    env_cfg.seed = _resolve_worker_seed(args.seed)
    robot_xml = resolve_g1_training_xml(args.robot_xml)
    if not robot_xml.is_file():
        raise FileNotFoundError(f"G1 training MuJoCo XML not found: {robot_xml}")
    env_cfg.scene.entities["robot"] = make_g1_training_robot_cfg(robot_xml)
    env_cfg.robot_xml = str(robot_xml)
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.motion_file is not None:
        env_cfg.commands["motion"].motion_file = args.motion_file
    validate_motion_file(env_cfg.commands["motion"].motion_file)
    if args.sampling_mode is not None:
        env_cfg.commands["motion"].sampling_mode = args.sampling_mode
    if args.rewind_prob is not None:
        env_cfg.commands["motion"].rewind_prob = args.rewind_prob
    if args.rewind_min_steps is not None:
        env_cfg.commands["motion"].rewind_min_steps = args.rewind_min_steps
    if args.rewind_max_steps is not None:
        env_cfg.commands["motion"].rewind_max_steps = args.rewind_max_steps
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name

    device = _resolve_device(args.device, torch)

    # Log directory (defined before env creation so video path is available)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    swanlab_active = _configure_experiment_logger(
        logger_name=args.logger,
        agent_cfg=agent_cfg,
        env_cfg=env_cfg,
        log_dir=log_dir,
    )

    # render_mode only needed for video recording
    render_mode = "rgb_array" if args.video else None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        if args.video:
            from mjlab.utils.wrappers import VideoRecorder
            env = VideoRecorder(
                env,
                video_folder=os.path.join(log_dir, "videos", "train"),
                step_trigger=lambda step: step % (args.video_interval * env_cfg.decimation) == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        RunnerCls = runner_cls if runner_cls is not None else MjlabOnPolicyRunner
        runner = RunnerCls(
            env,
            build_runner_cfg_dict(agent_cfg),
            log_dir=log_dir,
            device=device,
        )

        if args.resume is not None:
            print(f"[INFO] Resuming from: {args.resume}")
            runner.load(args.resume)
            print(
                f"[INFO] Running {agent_cfg.max_iterations} additional iterations "
                f"from checkpoint iteration {runner.current_learning_iteration}"
            )
        else:
            print(f"[INFO] Running {agent_cfg.max_iterations} iterations")

        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        print(f"[INFO] Rank {rank} interrupted; exiting gracefully.")
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        if swanlab_active:
            with contextlib.suppress(Exception):
                import swanlab

                swanlab.finish()
        _destroy_process_group(torch)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main(argv: Sequence[str] | None = None) -> None:
    cli_argv = list(sys.argv if argv is None else argv)
    parse_argv = cli_argv[1:]
    args = parse_args(parse_argv)

    if _should_launch_multi_gpu(args) or getattr(args, "all_gpus", False):
        import torch

        normalize_multi_gpu_args(args, torch)

    if _should_launch_multi_gpu(args):
        _launch_multi_gpu(args, cli_argv, num_envs=args.num_envs)
        return

    _run_worker(args)


if __name__ == "__main__":
    main()
