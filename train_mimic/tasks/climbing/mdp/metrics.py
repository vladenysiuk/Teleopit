"""Climbing episode metrics.

Per-step indicators use mjlab ``reduce="mean"`` / ``"last"``. Running maxima
are class-based terms (mjlab MetricsTermCfg has no ``reduce="max"`` yet).
Stateful counters that rewards already maintain are reported with ``last``.

``time_to_success`` keeps a per-env ``-1`` sentinel until first success, but
episode logging averages **successful envs only** (see
``install_time_to_success_success_only_aggregation``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height
from train_mimic.tasks.climbing.mdp.terminations import climbing_success_predicate

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.metrics_manager import MetricsTermCfg

_TTS_SUCCESS_ONLY_ATTR = "_teleopit_tts_success_only_reset"
TIME_TO_SUCCESS_TERM = "time_to_success"
TIME_TO_SUCCESS_LOG_KEY = f"Episode_Metrics/{TIME_TO_SUCCESS_TERM}"


def mean_successful_time_to_success(values: torch.Tensor) -> torch.Tensor:
    """Mean steps-to-success over entries with ``value >= 0``; NaN if none."""
    flat = values.reshape(-1).to(dtype=torch.float32)
    ok = flat >= 0
    if not bool(ok.any().item()):
        return torch.tensor(float("nan"), device=flat.device, dtype=torch.float32)
    return flat[ok].mean()


def correct_mean_time_to_success(
    contaminated_mean: float, success_rate: float, *, failure_sentinel: float = -1.0
) -> float:
    """Recover success-only mean from a mean that included ``failure_sentinel``.

    ``t' = (t - (1 - sr) * sentinel) / sr``. Returns NaN when ``sr == 0``.
    """
    if success_rate <= 0.0:
        return float("nan")
    return (contaminated_mean - (1.0 - success_rate) * failure_sentinel) / success_rate


def install_time_to_success_success_only_aggregation(env: ManagerBasedRlEnv) -> None:
    """Patch metrics reset so ``Episode_Metrics/time_to_success`` ignores failures.

    mjlab ``MetricsManager.reset`` does ``mean(_step_values[env_ids])`` for
    ``reduce="last"``. Unsuccessful envs still hold the ``-1`` sentinel, which
    pulls the logged mean down. Replace that scalar with the mean over
    successful envs only; omit the key when a reset batch has no successes.
    """
    mgr = getattr(env, "metrics_manager", None)
    if mgr is None or not hasattr(mgr, "active_terms"):
        return
    if TIME_TO_SUCCESS_TERM not in mgr.active_terms:
        return
    if getattr(mgr, _TTS_SUCCESS_ONLY_ATTR, False):
        return

    original_reset = mgr.reset

    def reset(env_ids: torch.Tensor | slice | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        idx = mgr.active_terms.index(TIME_TO_SUCCESS_TERM)
        vals = mgr._step_values[env_ids, idx].detach().clone()
        extras = original_reset(env_ids)
        if TIME_TO_SUCCESS_LOG_KEY not in extras:
            return extras
        reduced = mean_successful_time_to_success(vals)
        if torch.isfinite(reduced).item():
            extras[TIME_TO_SUCCESS_LOG_KEY] = reduced
        else:
            extras.pop(TIME_TO_SUCCESS_LOG_KEY, None)
        return extras

    mgr.reset = reset  # type: ignore[method-assign]
    setattr(mgr, _TTS_SUCCESS_ONLY_ATTR, True)


def _actuator_torque_saturated(env: ManagerBasedRlEnv, *, rtol: float = 0.98) -> torch.Tensor:
    robot = env.scene["robot"]
    force = getattr(robot.data, "actuator_force", None)
    if force is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    limits = getattr(robot.data, "actuator_force_limits", None)
    if limits is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.any(torch.abs(force) >= (limits * rtol), dim=-1)


class max_head_height_l:
    """Running maximum head / progress-body ladder-relative height (reduce=last).

    Body follows ``ClimbingRewardConfig.progress_body`` (default ``d435i_link``).
    """

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv) -> None:
        self._env = env
        self._body_name = str(cfg.params.get("body_name", "d435i_link"))
        self._max = torch.full(
            (env.num_envs,), float("-inf"), device=env.device, dtype=torch.float32
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # Managers construct before startup events; ladder may be absent at init.
        try:
            height = body_ladder_relative_height(self._env, body_name=self._body_name)
        except RuntimeError:
            self._max[env_ids] = float("-inf")
            return
        self._max[env_ids] = height[env_ids]

    def __call__(self, env: ManagerBasedRlEnv, *, body_name: str = "d435i_link") -> torch.Tensor:
        del body_name
        height = body_ladder_relative_height(env, body_name=self._body_name)
        self._max = torch.maximum(self._max, height)
        return self._max


def head_height_l(env: ManagerBasedRlEnv, *, body_name: str = "d435i_link") -> torch.Tensor:
    """Current head / progress-body ladder-relative height ``[B]``."""
    return body_ladder_relative_height(env, body_name=body_name)


def hand_contact_indicator(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mean hand-contact indicator this step ``[B]`` in ``[0, 1]``."""
    contacts = ClimbContactState.get(env)
    contacts.update()
    return contacts.state.hand_in_contact.to(dtype=torch.float32).mean(dim=-1)


def torque_saturation_indicator(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Any-actuator torque saturation indicator this step ``[B]``."""
    return _actuator_torque_saturated(env).to(dtype=torch.float32)


def valid_higher_attachments(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Count of paid higher-attachment events this episode ``[B]``."""
    return ClimbRewardState.get(env).valid_higher_attachment_count.to(dtype=torch.float32)


def invalid_latch_count(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Count of invalid latch attempts this episode ``[B]``."""
    return ClimbRewardState.get(env).invalid_latch_count.to(dtype=torch.float32)


def episode_success(
    env: ManagerBasedRlEnv,
    *,
    body_name: str = "d435i_link",
    pelvis_clearance_below_top_l: float,
    top_attach_margin_l: float,
    min_attached_hands: int,
) -> torch.Tensor:
    """Binary success flag for the current step ``[B]``."""
    return climbing_success_predicate(
        env,
        body_name=body_name,
        pelvis_clearance_below_top_l=pelvis_clearance_below_top_l,
        top_attach_margin_l=top_attach_margin_l,
        min_attached_hands=min_attached_hands,
    ).to(dtype=torch.float32)


def time_to_success_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Steps until first success, or ``-1`` if not yet successful ``[B]``.

    Per-env sentinel remains ``-1`` for debugging. Logged episode aggregates use
    successful envs only via ``install_time_to_success_success_only_aggregation``.
    """
    return ClimbRewardState.get(env).time_to_success_steps.to(dtype=torch.float32)
