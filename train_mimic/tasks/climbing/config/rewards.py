"""Climbing reward configuration."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from train_mimic.tasks.climbing.mdp import metrics as climb_metrics
from train_mimic.tasks.climbing.mdp import rewards as climb_rewards
from train_mimic.tasks.climbing.mdp import terminations as climb_terminations


@dataclass
class ClimbingRewardConfig:
    """Numeric parameters and reward-term weights for General-Climbing-G1."""

    progress_body: str = "pelvis"
    upward_progress_weight: float = 10.0
    new_attachment_weight: float = 2.0
    success_weight: float = 20.0
    # Success is relative to the sampled ladder: goal = top_rung_h - clearance.
    success_pelvis_clearance_below_top_l: float = 0.15
    success_top_attach_margin_l: float = 0.05
    success_min_attached_hands: int = 1
    attachment_height_eps: float = 1.0e-4
    time_penalty_coeff: float = 0.05
    action_rate_weight: float = 0.01
    effort_weight: float = 1.0e-4
    invalid_latch_weight: float = 0.5
    latch_overload_weight: float | None = None


@dataclass
class ClimbingTerminationConfig:
    """Termination thresholds for General-Climbing-G1."""

    progress_body: str = "pelvis"
    fall_min_height_w: float = 0.25
    fall_max_ladder_distance_xy: float = 1.5
    enable_nan_detection: bool = True
    enable_success_termination: bool = True
    enable_fall_termination: bool = True
    enable_latch_overload: bool = False
    latch_overload_force: float = 500.0


def configure_climbing_rewards(
    cfg: ManagerBasedRlEnvCfg,
    *,
    reward_cfg: ClimbingRewardConfig | None = None,
    term_cfg: ClimbingTerminationConfig | None = None,
) -> None:
    """Wire reward, termination, and metric terms into the env configuration."""
    reward_cfg = reward_cfg or ClimbingRewardConfig()
    term_cfg = term_cfg or ClimbingTerminationConfig()

    success_params = {
        "body_name": reward_cfg.progress_body,
        "pelvis_clearance_below_top_l": reward_cfg.success_pelvis_clearance_below_top_l,
        "top_attach_margin_l": reward_cfg.success_top_attach_margin_l,
        "min_attached_hands": reward_cfg.success_min_attached_hands,
    }
    fall_params = {
        "body_name": term_cfg.progress_body,
        "minimum_height_w": term_cfg.fall_min_height_w,
        "max_ladder_distance_xy": term_cfg.fall_max_ladder_distance_xy,
    }
    metric_body_params = {"body_name": reward_cfg.progress_body}

    cfg.rewards = {
        "upward_progress": RewardTermCfg(
            func=climb_rewards.upward_progress,
            weight=reward_cfg.upward_progress_weight,
            params={"body_name": reward_cfg.progress_body},
        ),
        "new_higher_attachment": RewardTermCfg(
            func=climb_rewards.new_higher_attachment,
            weight=reward_cfg.new_attachment_weight,
            params={"height_eps": reward_cfg.attachment_height_eps},
        ),
        "success": RewardTermCfg(
            func=climb_rewards.success_bonus,
            weight=reward_cfg.success_weight,
            params=dict(success_params),
        ),
        "time_penalty": RewardTermCfg(
            func=climb_rewards.time_penalty_mask,
            weight=-reward_cfg.time_penalty_coeff,
            params=dict(success_params),
        ),
        "action_rate": RewardTermCfg(
            func=mjlab_mdp.action_rate_l2,
            weight=-reward_cfg.action_rate_weight,
        ),
        "effort": RewardTermCfg(
            func=mjlab_mdp.joint_torques_l2,
            weight=-reward_cfg.effort_weight,
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "invalid_latch": RewardTermCfg(
            func=climb_rewards.invalid_latch_request,
            weight=-reward_cfg.invalid_latch_weight,
        ),
    }
    if reward_cfg.latch_overload_weight is not None:
        cfg.rewards["latch_overload"] = RewardTermCfg(
            func=climb_rewards.latch_overload_penalty,
            weight=-reward_cfg.latch_overload_weight,
        )

    cfg.terminations = {}
    if term_cfg.enable_success_termination:
        cfg.terminations["success"] = TerminationTermCfg(
            func=climb_terminations.climbing_success,
            params=dict(success_params),
        )
    if term_cfg.enable_fall_termination:
        cfg.terminations["fall"] = TerminationTermCfg(
            func=climb_terminations.fallen_or_far_from_ladder,
            params=fall_params,
        )
    cfg.terminations["time_out"] = TerminationTermCfg(func=mjlab_mdp.time_out, time_out=True)
    if term_cfg.enable_nan_detection:
        cfg.terminations["nan"] = TerminationTermCfg(func=mjlab_mdp.nan_detection)
    if term_cfg.enable_latch_overload:
        cfg.terminations["latch_overload"] = TerminationTermCfg(
            func=climb_terminations.latch_overload,
            params={"force_threshold": term_cfg.latch_overload_force},
        )

    # mjlab MetricsTermCfg supports reduce in {"mean", "last"} only; running
    # maxima use a class-based term reported with reduce="last".
    cfg.metrics = {
        "max_pelvis_height_l": MetricsTermCfg(
            func=climb_metrics.max_pelvis_height_l,
            reduce="last",
            params=dict(metric_body_params),
        ),
        "pelvis_height_l": MetricsTermCfg(
            func=climb_metrics.pelvis_height_l,
            reduce="last",
            params=dict(metric_body_params),
        ),
        "valid_higher_attachments": MetricsTermCfg(
            func=climb_metrics.valid_higher_attachments,
            reduce="last",
        ),
        "invalid_latch_count": MetricsTermCfg(
            func=climb_metrics.invalid_latch_count,
            reduce="last",
        ),
        "success": MetricsTermCfg(
            func=climb_metrics.episode_success,
            reduce="last",
            params=dict(success_params),
        ),
        "hand_contact_fraction": MetricsTermCfg(
            func=climb_metrics.hand_contact_indicator,
            reduce="mean",
        ),
        "time_to_success": MetricsTermCfg(
            func=climb_metrics.time_to_success_steps,
            reduce="last",
        ),
        "torque_saturation_fraction": MetricsTermCfg(
            func=climb_metrics.torque_saturation_indicator,
            reduce="mean",
        ),
    }
