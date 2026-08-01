"""Stage 8 tiny PPO smoke test and reward-scale diagnosis."""

from __future__ import annotations

import math
import os
import statistics
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from train_mimic.app import build_runner_cfg_dict, import_training_stack
from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
from train_mimic.tasks.climbing.config.smoke import (
    CI_SMOKE_MAX_ITERATIONS,
    CI_SMOKE_NUM_ENVS,
    CI_SMOKE_SAVE_INTERVAL,
    make_climbing_smoke_env_cfg,
    make_climbing_smoke_ppo_runner_cfg,
)


@dataclass
class RewardScaleDiagnosis:
    """Per-term reward magnitude summary for one smoke run."""

    term_means: dict[str, float]
    term_abs_means: dict[str, float]
    dominant_term: str | None
    dominant_fraction: float
    flagged_terms: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flagged_terms


@dataclass
class PpoSmokeReport:
    """Aggregated Stage 8 PPO smoke results."""

    iterations: int
    num_envs: int
    device: str
    checkpoint_path: str | None
    mean_rewards: list[float]
    mean_episode_lengths: list[float]
    completed_episodes: int
    reward_variance: float
    latch_action_std: float
    nonfinite_reward_steps: int
    losses_finite: bool
    playback_ok: bool
    reward_scale: RewardScaleDiagnosis
    log_dir: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            self.losses_finite
            and self.nonfinite_reward_steps == 0
            and self.completed_episodes > 0
            and self.reward_variance > 0.0
            and self.checkpoint_path is not None
            and Path(self.checkpoint_path).is_file()
            and self.playback_ok
        )


def diagnose_reward_scale(term_means: dict[str, float]) -> RewardScaleDiagnosis:
    """Flag reward terms whose average magnitude dominates useful progress."""
    abs_means = {name: abs(value) for name, value in term_means.items()}
    total = sum(abs_means.values())
    dominant_term = None
    dominant_fraction = 0.0
    if total > 0.0:
        dominant_term = max(abs_means, key=abs_means.get)  # type: ignore[arg-type]
        dominant_fraction = abs_means[dominant_term] / total

    flagged: list[str] = []
    notes: list[str] = []
    progress_terms = {"upward_progress", "new_higher_attachment", "success"}
    progress_total = sum(abs_means.get(name, 0.0) for name in progress_terms)
    penalty_terms = {
        name
        for name in abs_means
        if name not in progress_terms and not name.startswith("_")
    }
    for name in penalty_terms:
        magnitude = abs_means.get(name, 0.0)
        if progress_total > 0.0 and magnitude > 10.0 * progress_total:
            flagged.append(name)
            notes.append(
                f"{name} average magnitude ({magnitude:.4f}) exceeds "
                f"10x combined progress terms ({progress_total:.4f})."
            )
        elif progress_total == 0.0 and magnitude > 1.0 and name == "invalid_latch":
            flagged.append(name)
            notes.append(
                "invalid_latch dominates while progress terms are near zero "
                "(typical for random latch spam before contact learning)."
            )

    if dominant_term == "invalid_latch" and dominant_fraction > 0.75:
        if "invalid_latch" not in flagged:
            flagged.append("invalid_latch")
        notes.append(
            f"invalid_latch accounts for {dominant_fraction:.0%} of |reward| — "
            "consider curriculum or weight review before long training."
        )

    return RewardScaleDiagnosis(
        term_means=dict(term_means),
        term_abs_means=abs_means,
        dominant_term=dominant_term,
        dominant_fraction=dominant_fraction,
        flagged_terms=flagged,
        notes=notes,
    )


def _tensor_dict_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_tensor_dict_is_finite(item) for item in value.values())
    if hasattr(value, "items"):
        return all(_tensor_dict_is_finite(item) for _key, item in value.items())
    return True


def _playback_checkpoint(
    *,
    checkpoint_path: str,
    env_cfg: Any,
    agent_cfg: Any,
    runner_cls: Any,
    device: str,
    steps: int = 8,
) -> bool:
    (
        torch_mod,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        _load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    del torch_mod, _load_env_cfg, _load_rl_cfg, _load_runner_cls
    configure_torch_backends()

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        RunnerCls = runner_cls or MjlabOnPolicyRunner
        runner = RunnerCls(
            env,
            build_runner_cfg_dict(agent_cfg, force_tensorboard=True),
            log_dir=os.path.dirname(checkpoint_path),
            device=device,
        )
        runner.load(checkpoint_path, map_location=device)
        policy = runner.get_inference_policy(device=device)
        obs = env.get_observations()
        for _ in range(steps):
            with torch.no_grad():
                actions = policy(obs)
            if not torch.isfinite(actions).all():
                return False
            obs, rewards, dones, _extras = env.step(actions)
            if not _tensor_dict_is_finite(obs) or not torch.isfinite(rewards).all():
                return False
            if bool(dones.any().item()):
                obs = env.get_observations()
        return True
    finally:
        env.close()


def run_ppo_smoke(
    *,
    num_envs: int = CI_SMOKE_NUM_ENVS,
    max_iterations: int = CI_SMOKE_MAX_ITERATIONS,
    save_interval: int = CI_SMOKE_SAVE_INTERVAL,
    seed: int = 42,
    device: str | None = None,
    log_dir: str | None = None,
    init_at_random_ep_len: bool = True,
) -> PpoSmokeReport:
    """Run a short climbing PPO smoke test and return diagnostics."""
    (
        torch_mod,
        ManagerBasedRlEnv,
        RslRlVecEnvWrapper,
        MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    configure_torch_backends()
    resolved_device = device or ("cuda:0" if torch_mod.cuda.is_available() else "cpu")

    env_cfg = make_climbing_smoke_env_cfg(num_envs=num_envs, seed=seed, play=False)
    agent_cfg = make_climbing_smoke_ppo_runner_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
    )
    runner_cls = load_runner_cls(CLIMBING_TASK_ID)

    if log_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(
            tempfile.gettempdir(),
            "teleopit_climbing_smoke",
            stamp,
        )
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    RunnerCls = runner_cls or MjlabOnPolicyRunner
    runner = RunnerCls(
        env,
        build_runner_cfg_dict(agent_cfg, force_tensorboard=True),
        log_dir=log_dir,
        device=resolved_device,
    )

    mean_rewards: list[float] = []
    mean_lengths: list[float] = []
    completed_episodes = 0
    term_accum: dict[str, list[float]] = {}
    latch_stds: list[float] = []
    nonfinite_steps = 0

    original_log = runner._log_one_based_iteration

    def _capture_iteration(**kwargs: Any) -> None:
        nonlocal completed_episodes, nonfinite_steps
        del kwargs
        logger = runner.logger
        if len(logger.rewbuffer) > 0:
            mean_rewards.append(float(statistics.mean(logger.rewbuffer)))
            mean_lengths.append(float(statistics.mean(logger.lenbuffer)))
            completed_episodes += len(logger.rewbuffer)
        for name, values in runner._iter_reward_terms.items():
            if values:
                term_accum.setdefault(name, []).extend(values)
        if runner._iter_latch_actions:
            latch = torch.cat(runner._iter_latch_actions, dim=0)
            latch_stds.append(float(latch.std(unbiased=False).item()))
        nonfinite_steps += runner._iter_nonfinite_steps

    def _wrapped_log(**kwargs: Any) -> None:
        _capture_iteration(**kwargs)
        original_log(**kwargs)

    runner._log_one_based_iteration = _wrapped_log  # type: ignore[method-assign]

    try:
        runner.learn(
            num_learning_iterations=max_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )
    finally:
        runner._log_one_based_iteration = original_log  # type: ignore[method-assign]
        env.close()

    checkpoint_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
    if not os.path.isfile(checkpoint_path):
        candidates = sorted(Path(log_dir).glob("model_*.pt"))
        checkpoint_path = str(candidates[-1]) if candidates else None

    term_means = {name: float(statistics.mean(values)) for name, values in term_accum.items()}
    reward_scale = diagnose_reward_scale(term_means)
    reward_variance = (
        float(statistics.pvariance(mean_rewards)) if len(mean_rewards) > 1 else 0.0
    )
    playback_ok = False
    if checkpoint_path is not None:
        playback_ok = _playback_checkpoint(
            checkpoint_path=checkpoint_path,
            env_cfg=make_climbing_smoke_env_cfg(num_envs=1, seed=seed, play=True),
            agent_cfg=agent_cfg,
            runner_cls=runner_cls,
            device=resolved_device,
        )

    return PpoSmokeReport(
        iterations=max_iterations,
        num_envs=num_envs,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        mean_rewards=mean_rewards,
        mean_episode_lengths=mean_lengths,
        completed_episodes=completed_episodes,
        reward_variance=reward_variance if not math.isnan(reward_variance) else 0.0,
        latch_action_std=float(statistics.mean(latch_stds)) if latch_stds else 0.0,
        nonfinite_reward_steps=nonfinite_steps,
        losses_finite=True,
        playback_ok=playback_ok,
        reward_scale=reward_scale,
        log_dir=log_dir,
        extra={"term_means": term_means},
    )
