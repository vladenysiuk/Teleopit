"""Tests for train_mimic.distributed_launch multi-GPU helpers."""

from __future__ import annotations

import argparse

import pytest

from train_mimic.distributed_launch import (
    build_launcher_env,
    build_torchrun_command,
    filtered_argv_for_worker,
    normalize_multi_gpu_args,
    should_launch_multi_gpu,
)


class _CudaStub:
    _device_count = 4

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return _CudaStub._device_count


class _TorchStub:
    cuda = _CudaStub()


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "gpu_ids": None,
        "all_gpus": False,
        "master_port": 29500,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestDistributedLaunch:
    def test_normalize_all_gpus_populates_logical_ids(self) -> None:
        args = _args(all_gpus=True)
        normalize_multi_gpu_args(args, _TorchStub())
        assert args.gpu_ids == [0, 1, 2, 3]

    def test_normalize_all_gpus_requires_two_devices(self) -> None:
        _CudaStub._device_count = 1
        try:
            with pytest.raises(ValueError, match="at least 2 visible CUDA devices"):
                normalize_multi_gpu_args(_args(all_gpus=True), _TorchStub())
        finally:
            _CudaStub._device_count = 4

    def test_normalize_rejects_all_gpus_with_gpu_ids(self) -> None:
        with pytest.raises(ValueError, match="together"):
            normalize_multi_gpu_args(_args(all_gpus=True, gpu_ids=[0, 1]), _TorchStub())

    def test_should_launch_multi_gpu_with_all_gpus_flag(self) -> None:
        assert should_launch_multi_gpu(_args(all_gpus=True), env={"WORLD_SIZE": "1"}) is True
        assert should_launch_multi_gpu(_args(all_gpus=True), env={"WORLD_SIZE": "4"}) is False

    def test_build_launcher_env_all_gpus_respects_existing_mask(self) -> None:
        args = _args(all_gpus=True, gpu_ids=[0, 1])
        env = build_launcher_env(args, env={"CUDA_VISIBLE_DEVICES": "2,3", "PATH": "/bin"})
        assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert env["PATH"] == "/bin"

    def test_build_launcher_env_all_gpus_sets_visible_devices_when_unset(self) -> None:
        args = _args(all_gpus=True, gpu_ids=[0, 1, 2, 3])
        env = build_launcher_env(args, env={"PATH": "/bin"})
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    def test_filtered_argv_for_worker_strips_all_gpus(self) -> None:
        argv = [
            "train_climb.py",
            "--all_gpus",
            "--master_port",
            "29600",
            "--easy",
            "--num_envs",
            "256",
        ]
        assert filtered_argv_for_worker(argv) == [
            "train_climb.py",
            "--easy",
            "--num_envs",
            "256",
        ]

    def test_build_torchrun_command_with_resolved_all_gpus(self) -> None:
        args = _args(all_gpus=True, gpu_ids=[0, 1, 2, 3], master_port=29600)
        command = build_torchrun_command(
            args,
            ["train_climb.py", "--all_gpus", "--easy", "--num_envs", "256"],
        )
        assert command[:4] == [
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "--master_port=29600",
        ]
        assert command[4:] == ["train_climb.py", "--easy", "--num_envs", "256"]
