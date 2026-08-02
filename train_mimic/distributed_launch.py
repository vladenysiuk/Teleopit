"""Shared single-node multi-GPU launch helpers for training scripts."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import time
from argparse import Namespace
from typing import Sequence


def add_multi_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    """Register --gpu_ids, --all_gpus, and --master_port on a training CLI parser."""
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Single-node multi-GPU launch helper. Example: --gpu_ids 0 1 2 3. "
            "When multiple IDs are provided, the script relaunches itself via torchrun. "
            "Mutually exclusive with --all_gpus."
        ),
    )
    parser.add_argument(
        "--all_gpus",
        action="store_true",
        help=(
            "Use every visible CUDA device for single-node multi-GPU training (via torchrun). "
            "Respects an existing CUDA_VISIBLE_DEVICES mask. Mutually exclusive with --gpu_ids."
        ),
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for internal torchrun launch (default: 29500)",
    )


def normalize_multi_gpu_args(args: Namespace, torch_module: object) -> None:
    """Resolve --all_gpus into args.gpu_ids and validate launcher settings."""
    if getattr(args, "all_gpus", False) and args.gpu_ids:
        raise ValueError("Cannot use --all_gpus together with --gpu_ids")

    if getattr(args, "all_gpus", False):
        if not torch_module.cuda.is_available():
            raise RuntimeError("--all_gpus requires CUDA, but no CUDA device is available")
        count = int(torch_module.cuda.device_count())
        if count < 2:
            raise ValueError(
                f"--all_gpus requires at least 2 visible CUDA devices, found {count}"
            )
        args.gpu_ids = list(range(count))

    validate_multi_gpu_args(args)


def is_distributed_env(env: dict[str, str] | None = None) -> bool:
    runtime_env = os.environ if env is None else env
    return int(runtime_env.get("WORLD_SIZE", "1")) > 1


def should_launch_multi_gpu(args: Namespace, env: dict[str, str] | None = None) -> bool:
    if is_distributed_env(env):
        return False
    if getattr(args, "all_gpus", False):
        return True
    gpu_ids = list(args.gpu_ids or [])
    return len(gpu_ids) > 1


def validate_multi_gpu_args(args: Namespace) -> None:
    gpu_ids = list(args.gpu_ids or [])
    if getattr(args, "all_gpus", False) and gpu_ids:
        # normalize_multi_gpu_args populates gpu_ids; this catches manual Namespace misuse.
        pass
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"--gpu_ids contains duplicates: {gpu_ids}")
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"--gpu_ids must be non-negative, got {gpu_ids}")
    if args.master_port <= 0:
        raise ValueError(f"--master_port must be positive, got {args.master_port}")


def filtered_argv_for_worker(argv: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    skip_gpu_id_values = False
    skip_next_value = False
    for token in argv:
        if skip_gpu_id_values:
            if token.startswith("-"):
                skip_gpu_id_values = False
            else:
                continue
        if skip_next_value:
            skip_next_value = False
            continue

        if token == "--gpu_ids":
            skip_gpu_id_values = True
            continue
        if token.startswith("--gpu_ids="):
            continue
        if token == "--all_gpus":
            continue
        if token == "--master_port":
            skip_next_value = True
            continue
        if token.startswith("--master_port="):
            continue

        filtered.append(token)
    return filtered


def build_torchrun_command(args: Namespace, argv: Sequence[str]) -> list[str]:
    gpu_ids = list(args.gpu_ids or [])
    if len(gpu_ids) <= 1:
        raise ValueError("multi-GPU launch requires at least two gpu ids")

    worker_argv = filtered_argv_for_worker(argv)
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={len(gpu_ids)}",
        f"--master_port={args.master_port}",
        *worker_argv,
    ]


def build_launcher_env(args: Namespace, env: dict[str, str] | None = None) -> dict[str, str]:
    runtime_env = dict(os.environ if env is None else env)
    gpu_ids = list(args.gpu_ids or [])
    if getattr(args, "all_gpus", False):
        # Respect a pre-set visibility mask; workers already see logical cuda:0..N-1.
        if "CUDA_VISIBLE_DEVICES" not in runtime_env and gpu_ids:
            runtime_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    elif gpu_ids:
        runtime_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    return runtime_env


def wait_process(proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return proc.poll() is not None


def terminate_worker_group(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGINT)
    if wait_process(proc, timeout_s=5.0):
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGTERM)
    if wait_process(proc, timeout_s=5.0):
        return

    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    wait_process(proc, timeout_s=2.0)


def resolve_distributed_device(requested_device: str | None, torch_module: object) -> str:
    """Resolve cuda device; under torchrun each worker must use cuda:{LOCAL_RANK}."""
    if is_distributed_env():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        expected_device = f"cuda:{local_rank}"
        if requested_device is not None and requested_device != expected_device:
            raise ValueError(
                f"Distributed worker must use device '{expected_device}' for LOCAL_RANK={local_rank}, "
                f"got '{requested_device}'. Remove --device or set it to the local-rank device."
            )
        return expected_device

    if requested_device is not None:
        return requested_device
    return "cuda:0" if torch_module.cuda.is_available() else "cpu"


def resolve_worker_seed(base_seed: int, env: dict[str, str] | None = None) -> int:
    runtime_env = os.environ if env is None else env
    if not is_distributed_env(runtime_env):
        return base_seed
    global_rank = int(runtime_env.get("RANK", "0"))
    return base_seed + global_rank * 100003


def is_main_process(env: dict[str, str] | None = None) -> bool:
    runtime_env = os.environ if env is None else env
    return int(runtime_env.get("RANK", "0")) == 0


def launch_multi_gpu(
    args: Namespace,
    argv: Sequence[str],
    *,
    num_envs: int | None = None,
) -> None:
    validate_multi_gpu_args(args)
    command = build_torchrun_command(args, argv)
    env = build_launcher_env(args)
    per_gpu = num_envs if num_envs is not None else "(from preset/default)"
    label = "all visible GPUs" if getattr(args, "all_gpus", False) else list(args.gpu_ids or [])
    print(f"[INFO] Launching multi-GPU training on {label} with {per_gpu} envs/GPU")
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return_code = proc.wait()
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt received, stopping distributed workers...")
        terminate_worker_group(proc)
        raise SystemExit(130)

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def destroy_process_group(torch_module: object) -> None:
    distributed = getattr(torch_module, "distributed", None)
    if distributed is None or not distributed.is_available():
        return
    if not distributed.is_initialized():
        return
    with contextlib.suppress(Exception):
        distributed.destroy_process_group()
