"""Climbing event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.latch import LatchConfig
from train_mimic.tasks.climbing.config.observations import RelativeRungObsConfig
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.relative_rungs import RelativeRungProvider
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def reset_ladder_sample(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ladder_cfg: LadderConfig,
) -> None:
    """Resample ladder layout for selected environments."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    runtime = env.extras.get(LadderRuntime.EXTRA_KEY)
    if runtime is None:
        runtime = LadderRuntime.attach(env, ladder_cfg, seed=env.cfg.seed)
    runtime.resample(env_ids)


def attach_climb_contact_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ladder_cfg: LadderConfig,
) -> None:
    """Attach contact adapter after scene initialization."""
    del env_ids
    if ClimbContactState.EXTRA_KEY not in env.extras:
        ClimbContactState.attach(env, max_rungs=ladder_cfg.max_rungs)


def reset_climb_contact_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Clear cached hand contact state on reset."""
    adapter = env.extras.get(ClimbContactState.EXTRA_KEY)
    if adapter is not None:
        adapter.reset(env_ids)


def attach_climb_latch_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ladder_cfg: LadderConfig,
    latch_cfg: LatchConfig,
) -> None:
    """Attach latch adapter after scene initialization."""
    del env_ids
    if ClimbLatchState.EXTRA_KEY not in env.extras:
        ClimbLatchState.attach(env, ladder_cfg=ladder_cfg, latch_cfg=latch_cfg)


def attach_relative_rung_provider(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ladder_cfg: LadderConfig,
    relative_rung_cfg: RelativeRungObsConfig | None = None,
) -> None:
    """Attach relative-rung observation provider after scene initialization."""
    del env_ids
    if RelativeRungProvider.EXTRA_KEY not in env.extras:
        RelativeRungProvider.attach(
            env,
            ladder_cfg=ladder_cfg,
            obs_cfg=relative_rung_cfg or RelativeRungObsConfig(),
        )


def reset_climb_latch_state(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Deactivate latch constraints, then clear Python-side latch state.

    Must run before ``reset_ladder_sample`` so resampled sites cannot interact
    with a still-active connect equality for one physics step.
    """
    adapter = env.extras.get(ClimbLatchState.EXTRA_KEY)
    if adapter is not None:
        adapter.reset(env_ids)


def attach_climb_reward_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    progress_body: str = "pelvis",
) -> None:
    """Attach reward event-memory adapter after scene initialization."""
    del env_ids, progress_body
    if ClimbRewardState.EXTRA_KEY not in env.extras:
        ClimbRewardState.attach(env)


def reset_climb_reward_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    progress_body: str = "pelvis",
) -> None:
    """Clear per-episode reward memory on reset."""
    adapter = env.extras.get(ClimbRewardState.EXTRA_KEY)
    if adapter is not None:
        adapter.reset(env, env_ids, progress_body=progress_body)
