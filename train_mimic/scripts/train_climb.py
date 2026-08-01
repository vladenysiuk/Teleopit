#!/usr/bin/env python3
"""Train General-Climbing-G1 with mjlab + rsl_rl PPO.

Usage:
    # Stage 8 infrastructure smoke (64 envs, 100 iterations, fixed easy ladder)
    python train_mimic/scripts/train_climb.py --smoke

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
    resolve_device,
    validate_checkpoint_path,
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
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Stage 8 smoke preset: pinned easy ladder, 64 envs, 100 iterations, "
            "conservative episode length, no broad domain randomization."
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
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

    def _handle_shutdown(signum: int, _frame: Any) -> None:
        print(f"[INFO] Received signal {signum}, shutting down...")
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        raise KeyboardInterrupt

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    try:
        configure_torch_backends()

        _task_name, env_cfg, agent_cfg, runner_cls = load_task_components(
            CLIMBING_TASK_ID,
            load_env_cfg=load_env_cfg,
            load_rl_cfg=load_rl_cfg,
            load_runner_cls=load_runner_cls,
        )

        if args.smoke:
            num_envs = args.num_envs if args.num_envs is not None else SMOKE_NUM_ENVS
            max_iterations = (
                args.max_iterations if args.max_iterations is not None else SMOKE_MAX_ITERATIONS
            )
            env_cfg = make_climbing_smoke_env_cfg(
                num_envs=num_envs,
                seed=args.seed if args.seed != 42 else SMOKE_SEED,
                episode_length_s=SMOKE_EPISODE_LENGTH_S,
                play=False,
            )
            agent_cfg.max_iterations = max_iterations
            agent_cfg.save_interval = SMOKE_SAVE_INTERVAL
        else:
            if args.num_envs is not None:
                env_cfg.scene.num_envs = args.num_envs
            if args.max_iterations is not None:
                agent_cfg.max_iterations = args.max_iterations

        env_cfg.seed = args.seed
        if args.experiment_name is not None:
            agent_cfg.experiment_name = args.experiment_name
        elif agent_cfg.experiment_name != CLIMBING_EXPERIMENT_NAME:
            agent_cfg.experiment_name = CLIMBING_EXPERIMENT_NAME

        device = resolve_device(args.device, torch)
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
                f"[INFO] Starting {CLIMBING_TASK_ID}: "
                f"{env_cfg.scene.num_envs} envs, {agent_cfg.max_iterations} iterations"
            )

        start = time.time()
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        elapsed = time.time() - start
        print(f"[INFO] Training finished in {elapsed:.1f}s. Logs: {log_dir}")
    except KeyboardInterrupt:
        print("[INFO] Interrupted; exiting gracefully.")
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        if swanlab_active:
            with contextlib.suppress(Exception):
                import swanlab

                swanlab.finish()
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)


if __name__ == "__main__":
    main()
