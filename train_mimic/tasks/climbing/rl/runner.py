"""Climbing PPO runner with reward-term and latch-action diagnostics."""

from __future__ import annotations

import os
import statistics
from typing import Any

import torch
from rsl_rl.utils import check_nan

from train_mimic.tasks.climbing.mdp.metrics import TIME_TO_SUCCESS_LOG_KEY
from train_mimic.tasks.tracking.rl.runner import (
    MotionTrackingOnPolicyRunner,
    _one_based_iteration_range,
    _resolve_total_iterations,
)


class ClimbingOnPolicyRunner(MotionTrackingOnPolicyRunner):
    """General-Climbing-G1 runner with per-term reward and latch-action logging."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._iter_reward_terms: dict[str, list[float]] = {}
        self._iter_latch_actions: list[torch.Tensor] = []
        self._iter_nonfinite_steps = 0

    def _base_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env  # type: ignore[assignment]
        return env

    def _reset_iteration_diagnostics(self) -> None:
        self._iter_reward_terms = {}
        self._iter_latch_actions = []
        self._iter_nonfinite_steps = 0

    def _collapse_time_to_success_ep_extras(self) -> None:
        """Collapse success-only TTS batch means so parent logging stays finite.

        Reset batches with no successes omit ``Episode_Metrics/time_to_success``.
        The tracking logger only iterates keys from ``ep_extras[0]``, so fold all
        finite TTS batch means into a single entry on the first extras dict.
        """
        logger = self.logger
        if not logger.ep_extras:
            return
        values: list[float] = []
        for ep_info in logger.ep_extras:
            if TIME_TO_SUCCESS_LOG_KEY not in ep_info:
                continue
            raw = ep_info.pop(TIME_TO_SUCCESS_LOG_KEY)
            tensor = torch.as_tensor(raw, dtype=torch.float32).reshape(-1)
            if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all().item()):
                continue
            values.append(float(tensor.mean().item()))
        if not values:
            return
        logger.ep_extras[0][TIME_TO_SUCCESS_LOG_KEY] = torch.tensor(
            [sum(values) / len(values)], dtype=torch.float32
        )

    def _accumulate_step_diagnostics(
        self,
        actions: torch.Tensor,
        obs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        del obs, dones
        if self.cfg.get("check_for_nan", True):
            if not torch.isfinite(rewards).all():
                self._iter_nonfinite_steps += 1

        latch = actions[..., -2:]
        self._iter_latch_actions.append(latch.detach().cpu())

        base_env = self._base_env()
        reward_mgr = getattr(base_env, "reward_manager", None)
        if reward_mgr is None:
            return
        dt = float(getattr(base_env, "step_dt", 1.0))
        scale_dt = bool(getattr(getattr(base_env, "cfg", None), "scale_rewards_by_dt", True))
        for term_idx, name in enumerate(reward_mgr.active_terms):
            if name.startswith("_"):
                continue
            rate = float(reward_mgr._step_reward[:, term_idx].mean().item())
            contrib = rate * dt if scale_dt else rate
            self._iter_reward_terms.setdefault(name, []).append(contrib)

    def _log_climbing_diagnostics(self, it: int) -> None:
        writer = self.logger.writer
        if writer is None:
            return

        for name, values in self._iter_reward_terms.items():
            if not values:
                continue
            writer.add_scalar(f"RewardTerm/{name}", statistics.mean(values), it)

        if self._iter_latch_actions:
            latch = torch.cat(self._iter_latch_actions, dim=0)
            writer.add_scalar("Policy/latch_action_mean", float(latch.mean().item()), it)
            writer.add_scalar("Policy/latch_action_std", float(latch.std(unbiased=False).item()), it)
            writer.add_scalar(
                "Policy/latch_action_abs_mean", float(latch.abs().mean().item()), it
            )

        if self._iter_nonfinite_steps:
            writer.add_scalar("Train/nonfinite_reward_steps", self._iter_nonfinite_steps, it)

        policy = self.alg.get_policy()
        normalizer = getattr(policy, "obs_normalizer", None)
        if normalizer is not None and hasattr(normalizer, "_mean"):
            mean = normalizer._mean.detach()
            var = normalizer._var.detach()
            writer.add_scalar("ObsNorm/mean_abs", float(mean.abs().mean().item()), it)
            writer.add_scalar("ObsNorm/std_mean", float(var.sqrt().mean().item()), it)

        ladder_norms = getattr(policy, "obs_normalizers_ladder", None)
        if ladder_norms is not None:
            for group_name, module in ladder_norms.items():
                if hasattr(module, "_mean"):
                    writer.add_scalar(
                        f"ObsNorm/ladder_{group_name}_mean_abs",
                        float(module._mean.abs().mean().item()),
                        it,
                    )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run PPO with climbing-specific scalar logging."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = _resolve_total_iterations(start_it, num_learning_iterations)
        import time

        for it in _one_based_iteration_range(start_it, total_it):
            self._reset_iteration_diagnostics()
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    self._accumulate_step_diagnostics(actions, obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    )
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            for value in loss_dict.values():
                if not isinstance(value, (int, float)) or not (
                    value == value and abs(value) != float("inf")
                ):
                    raise RuntimeError(f"Non-finite PPO loss at iteration {it}: {loss_dict}")

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self._log_climbing_diagnostics(it)
            self._collapse_time_to_success_ep_extras()
            self._log_one_based_iteration(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(
                os.path.join(
                    self.logger.log_dir,
                    f"model_{self.current_learning_iteration}.pt",
                )
            )  # type: ignore[arg-type]
            self.logger.stop_logging_writer()
