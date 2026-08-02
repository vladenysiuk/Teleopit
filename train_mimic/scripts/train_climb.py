#!/usr/bin/env python3
"""Train General-Climbing-G1 with mjlab + rsl_rl PPO.

Usage:
    # Stage 8 infrastructure smoke (64 envs, 100 iterations, fixed easy ladder)
    python train_mimic/scripts/train_climb.py --smoke

    # Stage 9 easy learnability preset (256 envs/GPU, 800 iterations)
    python train_mimic/scripts/train_climb.py --easy

    # Single-node multi-GPU (num_envs is per GPU; 4x A100 -> 1024 total with 256/GPU)
    python train_mimic/scripts/train_climb.py --easy \
        --gpu_ids 0 1 2 3 --num_envs 256

    # Or use every visible GPU
    python train_mimic/scripts/train_climb.py --easy --all_gpus --num_envs 256

    # Custom run
    python train_mimic/scripts/train_climb.py \
        --num_envs 4096 --max_iterations 30000

    # Resume
    python train_mimic/scripts/train_climb.py \
        --resume logs/rsl_rl/g1_general_climbing/<run>/model_1000.pt \
        --max_iterations 5000
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Sequence

from train_mimic.app import (
    build_runner_cfg_dict,
    import_training_stack,
    load_task_components,
    validate_checkpoint_path,
)
from train_mimic.distributed_launch import (
    add_multi_gpu_arguments,
    destroy_process_group,
    is_main_process,
    launch_multi_gpu,
    normalize_multi_gpu_args,
    resolve_distributed_device,
    resolve_worker_seed,
    should_launch_multi_gpu,
)
from train_mimic.tasks.climbing.config.constants import (
    CLIMBING_EXPERIMENT_NAME,
    CLIMBING_TASK_ID,
)
from train_mimic.tasks.climbing.config.smoke import (
    SMOKE_EPISODE_LENGTH_S,
    SMOKE_MAX_ITERATIONS,
    SMOKE_NUM_ENVS,
    SMOKE_SAVE_INTERVAL,
    SMOKE_SEED,
    make_climbing_smoke_env_cfg,
)
from train_mimic.tasks.climbing.config.easy import (
    EASY_EPISODE_LENGTH_S,
    EASY_MAX_ITERATIONS,
    EASY_NUM_ENVS,
    EASY_SAVE_INTERVAL,
    EASY_SEED,
    make_climbing_easy_env_cfg,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train G1 climbing policy (mjlab).")
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "swanlab"],
    )
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    add_multi_gpu_arguments(parser)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Stage 8 smoke preset: pinned easy ladder, 64 envs, 100 iterations, "
            "conservative episode length, no broad domain randomization."
        ),
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help=(
            "Stage 9 easy learnability preset: assisted hold start, both hands "
            "attached, fixed ladder, 256 envs/GPU, 800 iterations."
        ),
    )
    return parser.parse_args(argv)


def _configure_experiment_logger(
    *,
    logger_name: str,
    agent_cfg: Any,
    env_cfg: Any,
    log_dir: str,
) -> bool:
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
    if not is_main_process():
        return False

    try:
        import swanlab
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "swanlab package is required for --logger swanlab. Install it with `pip install swanlab`."
        ) from exc

    swanlab.init(
        project=agent_cfg.experiment_name,
        name=os.path.basename(log_dir),
        log_dir=log_dir,
        config={
            "experiment_name": agent_cfg.experiment_name,
            "num_envs": env_cfg.scene.num_envs,
            "max_iterations": agent_cfg.max_iterations,
            "episode_length_s": env_cfg.episode_length_s,
            "task": CLIMBING_TASK_ID,
        },
    )
    swanlab.sync_tensorboard_torch(types=["scalar", "scalars", "image", "text"])
    return True


def _build_env_cfg(args: argparse.Namespace) -> tuple[Any, int, int | None, int | None, int]:
    """Return env_cfg, num_envs, max_iterations, save_interval, base_seed."""
    if args.smoke:
        num_envs = args.num_envs if args.num_envs is not None else SMOKE_NUM_ENVS
        max_iterations = (
            args.max_iterations if args.max_iterations is not None else SMOKE_MAX_ITERATIONS
        )
        seed = args.seed if args.seed != 42 else SMOKE_SEED
        env_cfg = make_climbing_smoke_env_cfg(
            num_envs=num_envs,
            seed=seed,
            episode_length_s=SMOKE_EPISODE_LENGTH_S,
            play=False,
        )
        return env_cfg, num_envs, max_iterations, SMOKE_SAVE_INTERVAL, seed

    if args.easy:
        num_envs = args.num_envs if args.num_envs is not None else EASY_NUM_ENVS
        max_iterations = (
            args.max_iterations if args.max_iterations is not None else EASY_MAX_ITERATIONS
        )
        seed = args.seed if args.seed != 42 else EASY_SEED
        env_cfg = make_climbing_easy_env_cfg(
            num_envs=num_envs,
            seed=seed,
            episode_length_s=EASY_EPISODE_LENGTH_S,
            play=False,
        )
        return env_cfg, num_envs, max_iterations, EASY_SAVE_INTERVAL, seed

    _task_name, env_cfg, _agent_cfg, _runner_cls = load_task_components(CLIMBING_TASK_ID)
    num_envs = args.num_envs if args.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.scene.num_envs = num_envs
    return env_cfg, num_envs, args.max_iterations, None, args.seed


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
    swanlab_active = False
    rank = os.environ.get("RANK", "0")

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        print(f"[INFO] Rank {rank} received signal {signum}, shutting down...")
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        destroy_process_group(torch)
        raise KeyboardInterrupt

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        configure_torch_backends()

        _task_name, _base_env_cfg, agent_cfg, runner_cls = load_task_components(
            CLIMBING_TASK_ID,
            load_env_cfg=load_env_cfg,
            load_rl_cfg=load_rl_cfg,
            load_runner_cls=load_runner_cls,
        )

        env_cfg, num_envs, preset_max_iters, preset_save_interval, base_seed = _build_env_cfg(args)
        env_cfg.seed = resolve_worker_seed(base_seed)

        if preset_max_iters is not None:
            agent_cfg.max_iterations = preset_max_iters
        elif args.max_iterations is not None:
            agent_cfg.max_iterations = args.max_iterations

        if preset_save_interval is not None:
            agent_cfg.save_interval = preset_save_interval

        if args.experiment_name is not None:
            agent_cfg.experiment_name = args.experiment_name
        elif agent_cfg.experiment_name != CLIMBING_EXPERIMENT_NAME:
            agent_cfg.experiment_name = CLIMBING_EXPERIMENT_NAME

        device = resolve_distributed_device(args.device, torch)

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

        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        RunnerCls = runner_cls or MjlabOnPolicyRunner
        runner = RunnerCls(
            env,
            build_runner_cfg_dict(agent_cfg),
            log_dir=log_dir,
            device=device,
        )

        if args.resume is not None:
            validate_checkpoint_path(args.resume)
            print(f"[INFO] Resuming from: {args.resume}")
            runner.load(args.resume)
            print(
                f"[INFO] Running {agent_cfg.max_iterations} additional iterations "
                f"from checkpoint iteration {runner.current_learning_iteration}"
            )
        else:
            print(
                f"[INFO] Starting {CLIMBING_TASK_ID} rank {rank}: "
                f"{num_envs} envs/GPU, {agent_cfg.max_iterations} iterations, device={device}"
            )

        start = time.time()
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        elapsed = time.time() - start
        if is_main_process():
            print(f"[INFO] Training finished in {elapsed:.1f}s. Logs: {log_dir}")
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
        destroy_process_group(torch)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


def main(argv: Sequence[str] | None = None) -> None:
    cli_argv = list(sys.argv if argv is None else argv)
    args = parse_args(cli_argv[1:])

    if should_launch_multi_gpu(args) or getattr(args, "all_gpus", False):
        import torch

        normalize_multi_gpu_args(args, torch)

    if should_launch_multi_gpu(args):
        launch_multi_gpu(args, cli_argv, num_envs=args.num_envs)
        return

    _run_worker(args)


if __name__ == "__main__":
    main()
