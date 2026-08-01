"""Stage 9 learnability experiment: baselines, short PPO, and metric comparison."""

from __future__ import annotations

import os
import statistics
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

from train_mimic.app import build_runner_cfg_dict, import_training_stack
from train_mimic.tasks.climbing.config.constants import CLIMBING_TASK_ID
from train_mimic.tasks.climbing.config.easy import (
    CI_EASY_MAX_ITERATIONS,
    CI_EASY_NUM_ENVS,
    CI_EASY_SAVE_INTERVAL,
    make_climbing_easy_env_cfg,
    make_climbing_easy_ppo_runner_cfg,
)
from train_mimic.tasks.climbing.debug_mdp import (
    make_random_action,
    make_zero_action,
    run_agent_rollout,
)
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height
from train_mimic.tasks.climbing.mdp.metrics import _actuator_torque_saturated
from train_mimic.tasks.climbing.ppo_smoke import _playback_checkpoint, diagnose_reward_scale


@dataclass
class EpisodeRolloutMetrics:
    """High-level climbing metrics aggregated from one rollout."""

    agent: str
    steps: int
    completed_episodes: int
    max_pelvis_height_l: float
    max_valid_higher_attachments: int
    max_invalid_latch: int
    invalid_latch_per_step: float
    torque_saturation_fraction: float
    reward_sum: float
    nonfinite_obs_steps: int = 0
    nonfinite_sim_steps: int = 0

    @property
    def ok(self) -> bool:
        return self.nonfinite_obs_steps == 0 and self.nonfinite_sim_steps == 0


@dataclass
class LearnabilityComparison:
    """Policy vs baseline metric deltas."""

    pelvis_height_gain_vs_zero: float
    pelvis_height_gain_vs_random: float
    attachment_gain_vs_zero: int
    attachment_gain_vs_random: int
    invalid_latch_reduction_vs_zero: float
    invalid_latch_reduction_vs_random: float
    torque_saturation_delta_vs_zero: float
    notes: list[str] = field(default_factory=list)

    @property
    def learning_signal(self) -> bool:
        """Conservative pre-run success evidence from Stage 9 plan."""
        return (
            self.pelvis_height_gain_vs_zero > 0.0
            or self.pelvis_height_gain_vs_random > 0.0
            or self.attachment_gain_vs_zero > 0
            or self.attachment_gain_vs_random > 0
        )


@dataclass
class LearnabilityReport:
    """Aggregated Stage 9 easy experiment results."""

    seed: int
    num_envs: int
    train_iterations: int
    device: str
    zero: EpisodeRolloutMetrics
    random: EpisodeRolloutMetrics
    trained: EpisodeRolloutMetrics | None
    comparison: LearnabilityComparison | None
    checkpoint_path: str | None
    log_dir: str
    reward_scale_flags: list[str] = field(default_factory=list)
    playback_ok: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        base_ok = self.zero.ok and self.random.ok
        if self.trained is None:
            return base_ok
        return base_ok and self.trained.ok and self.playback_ok


def _run_vec_policy_rollout(vec_env, policy, *, num_steps: int):
    """Roll out a loaded policy through ``RslRlVecEnvWrapper``."""
    from train_mimic.tasks.climbing.debug_mdp import MdpRolloutSummary

    base_env = _unwrap_env(vec_env)
    summary = MdpRolloutSummary()
    base_env.reset()
    obs = vec_env.get_observations()

    for step_idx in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, extras = vec_env.step(actions)
        summary.steps += 1
        summary.reward_sum += float(rewards.sum().item())
        if not torch.isfinite(rewards).all():
            summary.nonfinite_sim_steps += 1
        dones_bool = dones.to(torch.bool)
        if bool(dones_bool.any().item()):
            summary.completed_episodes += int(dones_bool.sum().item())
            obs = vec_env.get_observations()
    return summary


def _collect_torque_saturation_fraction(env, num_steps: int) -> float:
    total = 0
    saturated = 0
    for _ in range(num_steps):
        sat = _actuator_torque_saturated(env)
        total += env.num_envs
        saturated += int(sat.sum().item())
        env.step(make_zero_action(env))
    return saturated / max(total, 1)


def _unwrap_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env  # type: ignore[assignment]
    return base


def collect_rollout_metrics(
    env,
    action_fn: Callable,
    *,
    agent: str,
    num_steps: int,
) -> EpisodeRolloutMetrics:
    """Run a headless rollout and aggregate climbing metrics."""
    base_env = _unwrap_env(env)
    base_env.reset()
    summary = run_agent_rollout(base_env, action_fn, num_steps=num_steps)

    pelvis_h = body_ladder_relative_height(base_env, body_name="pelvis")
    max_pelvis = float(pelvis_h.max().item())

    reward_state = ClimbRewardState.get(base_env)
    max_attach = int(reward_state.valid_higher_attachment_count.max().item())
    max_invalid = int(reward_state.invalid_latch_count.max().item())

    probe_steps = min(8, max(1, num_steps // 10))
    sat_frac = _collect_torque_saturation_fraction(base_env, probe_steps)

    return EpisodeRolloutMetrics(
        agent=agent,
        steps=summary.steps,
        completed_episodes=summary.completed_episodes,
        max_pelvis_height_l=max_pelvis,
        max_valid_higher_attachments=max_attach,
        max_invalid_latch=max_invalid,
        invalid_latch_per_step=max_invalid / max(summary.steps, 1),
        torque_saturation_fraction=sat_frac,
        reward_sum=summary.reward_sum,
        nonfinite_obs_steps=summary.nonfinite_obs_steps,
        nonfinite_sim_steps=summary.nonfinite_sim_steps,
    )


def compare_learnability(
    *,
    zero: EpisodeRolloutMetrics,
    random: EpisodeRolloutMetrics,
    trained: EpisodeRolloutMetrics,
) -> LearnabilityComparison:
    """Compare trained policy metrics against zero/random baselines."""
    notes: list[str] = []
    if trained.max_pelvis_height_l <= zero.max_pelvis_height_l:
        notes.append("trained max pelvis height did not exceed zero baseline")
    if trained.max_invalid_latch >= zero.max_invalid_latch:
        notes.append("trained invalid latch count did not improve vs zero")
    if trained.torque_saturation_fraction > zero.torque_saturation_fraction + 0.15:
        notes.append("torque saturation increased vs zero baseline")

    return LearnabilityComparison(
        pelvis_height_gain_vs_zero=trained.max_pelvis_height_l - zero.max_pelvis_height_l,
        pelvis_height_gain_vs_random=trained.max_pelvis_height_l - random.max_pelvis_height_l,
        attachment_gain_vs_zero=trained.max_valid_higher_attachments - zero.max_valid_higher_attachments,
        attachment_gain_vs_random=trained.max_valid_higher_attachments - random.max_valid_higher_attachments,
        invalid_latch_reduction_vs_zero=zero.invalid_latch_per_step - trained.invalid_latch_per_step,
        invalid_latch_reduction_vs_random=random.invalid_latch_per_step - trained.invalid_latch_per_step,
        torque_saturation_delta_vs_zero=trained.torque_saturation_fraction - zero.torque_saturation_fraction,
        notes=notes,
    )


def run_baseline_comparison(
    *,
    num_envs: int = CI_EASY_NUM_ENVS,
    seed: int = 42,
    rollout_steps: int = 120,
    device: str | None = None,
) -> tuple[EpisodeRolloutMetrics, EpisodeRolloutMetrics]:
    """Run zero and random baselines on the easy env."""
    (
        torch_mod,
        ManagerBasedRlEnv,
        _RslRlVecEnvWrapper,
        _MjlabOnPolicyRunner,
        _load_env_cfg,
        _load_rl_cfg,
        _load_runner_cls,
        configure_torch_backends,
    ) = import_training_stack()
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401

    configure_torch_backends()
    resolved_device = device or ("cuda:0" if torch_mod.cuda.is_available() else "cpu")

    env_cfg = make_climbing_easy_env_cfg(num_envs=num_envs, seed=seed, play=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
    try:
        zero = collect_rollout_metrics(
            env,
            lambda _env, _step: make_zero_action(_env),
            agent="zero",
            num_steps=rollout_steps,
        )
        env.reset()
        gen = torch.Generator(device=env.device)
        gen.manual_seed(seed)
        random = collect_rollout_metrics(
            env,
            lambda _env, _step: make_random_action(_env, generator=gen),
            agent="random",
            num_steps=rollout_steps,
        )
        return zero, random
    finally:
        env.close()


def run_learnability_experiment(
    *,
    num_envs: int = CI_EASY_NUM_ENVS,
    max_iterations: int = CI_EASY_MAX_ITERATIONS,
    save_interval: int = CI_EASY_SAVE_INTERVAL,
    seed: int = 42,
    rollout_steps: int = 120,
    device: str | None = None,
    log_dir: str | None = None,
) -> LearnabilityReport:
    """Train on the easy preset, then compare against zero/random baselines."""
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

    zero, random = run_baseline_comparison(
        num_envs=num_envs,
        seed=seed,
        rollout_steps=rollout_steps,
        device=resolved_device,
    )

    env_cfg = make_climbing_easy_env_cfg(num_envs=num_envs, seed=seed, play=False)
    agent_cfg = make_climbing_easy_ppo_runner_cfg(
        max_iterations=max_iterations,
        save_interval=save_interval,
    )
    runner_cls = load_runner_cls(CLIMBING_TASK_ID)

    if log_dir is None:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(tempfile.gettempdir(), "teleopit_climbing_easy", stamp)
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

    term_accum: dict[str, list[float]] = {}
    original_log = runner._log_one_based_iteration

    def _capture(**kwargs: Any) -> None:
        del kwargs
        for name, values in runner._iter_reward_terms.items():
            if values:
                term_accum.setdefault(name, []).extend(values)

    def _wrapped_log(**kwargs: Any) -> None:
        _capture(**kwargs)
        original_log(**kwargs)

    runner._log_one_based_iteration = _wrapped_log  # type: ignore[method-assign]
    try:
        runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
    finally:
        runner._log_one_based_iteration = original_log  # type: ignore[method-assign]
        env.close()

    checkpoint_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
    if not os.path.isfile(checkpoint_path):
        candidates = sorted(Path(log_dir).glob("model_*.pt"))
        checkpoint_path = str(candidates[-1]) if candidates else None

    term_means = {name: float(statistics.mean(values)) for name, values in term_accum.items()}
    reward_scale = diagnose_reward_scale(term_means)

    playback_ok = False
    trained: EpisodeRolloutMetrics | None = None
    comparison: LearnabilityComparison | None = None

    if checkpoint_path is not None:
        playback_ok = _playback_checkpoint(
            checkpoint_path=checkpoint_path,
            env_cfg=make_climbing_easy_env_cfg(num_envs=num_envs, seed=seed, play=True),
            agent_cfg=agent_cfg,
            runner_cls=runner_cls,
            device=resolved_device,
        )

        play_env = ManagerBasedRlEnv(
            cfg=make_climbing_easy_env_cfg(num_envs=num_envs, seed=seed, play=True),
            device=resolved_device,
        )
        vec_env = RslRlVecEnvWrapper(play_env, clip_actions=agent_cfg.clip_actions)
        try:
            runner_eval = RunnerCls(
                vec_env,
                build_runner_cfg_dict(agent_cfg, force_tensorboard=True),
                log_dir=log_dir,
                device=resolved_device,
            )
            runner_eval.load(checkpoint_path, map_location=resolved_device)
            policy = runner_eval.get_inference_policy(device=resolved_device)

            base_env = _unwrap_env(vec_env)
            base_env.reset()
            summary = _run_vec_policy_rollout(vec_env, policy, num_steps=rollout_steps)
            pelvis_h = body_ladder_relative_height(base_env, body_name="pelvis")
            reward_state = ClimbRewardState.get(base_env)
            trained = EpisodeRolloutMetrics(
                agent="trained",
                steps=summary.steps,
                completed_episodes=summary.completed_episodes,
                max_pelvis_height_l=float(pelvis_h.max().item()),
                max_valid_higher_attachments=int(
                    reward_state.valid_higher_attachment_count.max().item()
                ),
                max_invalid_latch=int(reward_state.invalid_latch_count.max().item()),
                invalid_latch_per_step=int(reward_state.invalid_latch_count.max().item())
                / max(summary.steps, 1),
                torque_saturation_fraction=_collect_torque_saturation_fraction(
                    base_env, min(8, max(1, rollout_steps // 10))
                ),
                reward_sum=summary.reward_sum,
                nonfinite_obs_steps=summary.nonfinite_obs_steps,
                nonfinite_sim_steps=summary.nonfinite_sim_steps,
            )
            comparison = compare_learnability(zero=zero, random=random, trained=trained)
        finally:
            vec_env.close()

    return LearnabilityReport(
        seed=seed,
        num_envs=num_envs,
        train_iterations=max_iterations,
        device=resolved_device,
        zero=zero,
        random=random,
        trained=trained,
        comparison=comparison,
        checkpoint_path=checkpoint_path,
        log_dir=log_dir,
        reward_scale_flags=list(reward_scale.flagged_terms),
        playback_ok=playback_ok,
        extra={"term_means": term_means, "reward_scale_notes": reward_scale.notes},
    )


def print_baseline_report(zero: EpisodeRolloutMetrics, random: EpisodeRolloutMetrics) -> None:
    for metrics in (zero, random):
        print(f"[baseline {metrics.agent}]")
        print(f"  max_pelvis_height_l={metrics.max_pelvis_height_l:.4f}")
        print(f"  max_valid_higher_attachments={metrics.max_valid_higher_attachments}")
        print(f"  max_invalid_latch={metrics.max_invalid_latch}")
        print(f"  invalid_latch_per_step={metrics.invalid_latch_per_step:.4f}")
        print(f"  torque_saturation_fraction={metrics.torque_saturation_fraction:.4f}")
        print(f"  completed_episodes={metrics.completed_episodes}")


def print_learnability_report(report: LearnabilityReport) -> None:
    print("[learnability] device:", report.device)
    print("[learnability] log_dir:", report.log_dir)
    print("[learnability] checkpoint:", report.checkpoint_path)
    print_baseline_report(report.zero, report.random)
    if report.trained is not None:
        print("[baseline trained]")
        print(f"  max_pelvis_height_l={report.trained.max_pelvis_height_l:.4f}")
        print(f"  max_valid_higher_attachments={report.trained.max_valid_higher_attachments}")
        print(f"  max_invalid_latch={report.trained.max_invalid_latch}")
        print(f"  playback_ok={report.playback_ok}")
    if report.comparison is not None:
        cmp = report.comparison
        print("[learnability comparison]")
        print(f"  pelvis_gain_vs_zero={cmp.pelvis_height_gain_vs_zero:.4f}")
        print(f"  pelvis_gain_vs_random={cmp.pelvis_height_gain_vs_random:.4f}")
        print(f"  attachment_gain_vs_zero={cmp.attachment_gain_vs_zero}")
        print(f"  invalid_latch_reduction_vs_zero={cmp.invalid_latch_reduction_vs_zero:.4f}")
        print(f"  learning_signal={cmp.learning_signal}")
        for note in cmp.notes:
            print("[learnability note]", note)
    if report.reward_scale_flags:
        print("[learnability] reward_scale_flags:", ", ".join(report.reward_scale_flags))
