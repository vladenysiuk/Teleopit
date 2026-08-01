"""Stage 7 MDP rollout helpers: agents, reset stress, and capacity checks."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from train_mimic.tasks.climbing.config.ladder import LadderConfig
from train_mimic.tasks.climbing.config.hold_pose import HoldPoseConfig
from train_mimic.tasks.climbing.debug_hold_pose import (
    HoldPoseSolution,
    build_hold_action,
    initialize_hold_pose_scene,
    run_headless_hold,
)
from train_mimic.tasks.climbing.debug_latch import build_latch_action
from train_mimic.tasks.climbing.debug_probe import nudge_robot_away_from_ladder
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime
from train_mimic.tasks.climbing.mdp.common import body_ladder_relative_height
from train_mimic.tasks.climbing.reset_state import (
    assert_carryover_preserved,
    assert_env_unchanged,
    assert_reset_env_clean,
    capture_reset_state,
    history_is_post_reset,
    ladder_resampled_across_resets,
    pollute_reset_sensitive_state,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

AgentKind = Literal["zero", "random", "scripted"]
ScriptPhaseKind = Literal[
    "hold",
    "attach",
    "detach",
    "invalid_attach",
    "trigger_fall",
    "neutral",
]


def _iter_obs_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_obs_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_obs_tensors(item)


def observations_finite(env: ManagerBasedRlEnv) -> bool:
    """Return True when every observation tensor is finite."""
    obs = env.observation_manager.compute()
    for tensor in _iter_obs_tensors(obs):
        if not torch.isfinite(tensor).all():
            return False
    return True


def randomized_ladder_cfg() -> LadderConfig:
    """Registered-task ladder randomization ranges for reset-stress tests."""
    return LadderConfig(
        max_rungs=12,
        min_active_rungs=4,
        max_active_rungs=10,
        spacing_min=0.22,
        spacing_max=0.28,
        ladder_distance_range=(0.60, 1.00),
        ladder_yaw_range=(-0.30, 0.30),
        ladder_tilt_range=(-0.05, 0.05),
    )


def pinned_ladder_cfg() -> LadderConfig:
    """Fixed geometry for reproducibility rollouts."""
    return LadderConfig(
        max_rungs=8,
        min_active_rungs=6,
        max_active_rungs=6,
        spacing_min=0.25,
        spacing_max=0.25,
        ladder_distance_range=(0.85, 0.85),
        ladder_yaw_range=(0.0, 0.0),
        ladder_tilt_range=(0.0, 0.0),
    )


def simulation_capacity_snapshot(env: ManagerBasedRlEnv) -> dict[str, int]:
    """Read current contact/constraint counts from simulator data."""
    data = env.sim.data
    nacon_total = int(data.nacon.reshape(-1)[0].item())
    nefc = data.nefc
    max_nefc = int(nefc.max().item()) if hasattr(nefc, "max") else int(nefc)
    nconmax = int(env.cfg.sim.nconmax)
    njmax = int(env.cfg.sim.njmax)
    latch = env.extras.get(ClimbLatchState.EXTRA_KEY)
    active_eq = 0
    max_eq_per_env = 0
    if latch is not None:
        eq_active = data.eq_active.to(torch.int64)
        active_eq = int(eq_active.sum().item())
        max_eq_per_env = int(eq_active.sum(dim=-1).max().item())
    # Global nacon; approximate per-env upper bound when batched.
    nacon_per_env_upper = (nacon_total + max(env.num_envs - 1, 0)) // max(env.num_envs, 1)
    return {
        "nacon_total": nacon_total,
        "nacon_per_env_upper": nacon_per_env_upper,
        "max_nefc": max_nefc,
        "active_equalities": active_eq,
        "max_equalities_per_env": max_eq_per_env,
        "ncon_headroom": nconmax * env.num_envs - nacon_total,
        "njmax_headroom": njmax - max_nefc,
    }


def capacity_within_limits(env: ManagerBasedRlEnv, cap: dict[str, int]) -> bool:
    """Return True when batched contact/constraint counts stay within cfg budgets."""
    ncon_budget = int(env.cfg.sim.nconmax) * env.num_envs
    njmax = int(env.cfg.sim.njmax)
    return cap["nacon_total"] < ncon_budget and cap["max_nefc"] < njmax


def make_zero_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Zero-action policy matching mjlab play --agent zero."""
    return torch.zeros(
        env.num_envs,
        env.action_manager.total_action_dim,
        device=env.device,
        dtype=torch.float32,
    )


def make_random_action(
    env: ManagerBasedRlEnv,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniform random action in [-1, 1] matching mjlab play --agent random."""
    return (
        2.0 * torch.rand(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
            dtype=torch.float32,
            generator=generator,
        )
        - 1.0
    )


def rollout_signature(env: ManagerBasedRlEnv, *, env_idx: int = 0) -> tuple[float, ...]:
    """Compact high-level state signature for reproducibility checks."""
    pelvis_h = float(body_ladder_relative_height(env, body_name="pelvis")[env_idx].item())
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    attached = tuple(int(v) for v in latch.state.attached[env_idx].tolist())
    rung_ids = tuple(int(v) for v in latch.state.rung_id[env_idx].tolist())
    reward_state = ClimbRewardState.get(env)
    higher = int(reward_state.valid_higher_attachment_count[env_idx].item())
    invalid = int(reward_state.invalid_latch_count[env_idx].item())
    return (pelvis_h, *attached, *rung_ids, higher, invalid)


@dataclass
class MdpRolloutSummary:
    """Aggregated statistics from a headless MDP rollout."""

    steps: int = 0
    episode_resets: int = 0
    completed_episodes: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    nonfinite_obs_steps: int = 0
    nonfinite_sim_steps: int = 0
    max_nacon: int = 0
    max_nefc: int = 0
    max_nacon_per_env_upper: int = 0
    max_equalities_per_env: int = 0
    min_ncon_headroom: int = 1_000_000_000
    min_njmax_headroom: int = 1_000_000_000
    max_active_equalities: int = 0
    termination_counts: dict[str, int] = field(default_factory=dict)
    reward_sum: float = 0.0
    reward_term_sums: dict[str, float] = field(default_factory=dict)
    max_valid_attachments: int = 0
    max_invalid_latch: int = 0
    signatures: list[tuple[float, ...]] = field(default_factory=list)


def _record_terminations(
    env: ManagerBasedRlEnv,
    summary: MdpRolloutSummary,
    *,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    extras: dict[str, Any] | None,
) -> None:
    if bool(terminated.any().item()):
        summary.termination_counts["terminated"] = summary.termination_counts.get(
            "terminated", 0
        ) + int(terminated.sum().item())
    if bool(truncated.any().item()):
        summary.termination_counts["truncated"] = summary.termination_counts.get(
            "truncated", 0
        ) + int(truncated.sum().item())
    if isinstance(extras, dict) and "time_outs" in extras:
        timeouts = extras["time_outs"]
        if isinstance(timeouts, torch.Tensor) and bool(timeouts.any().item()):
            summary.termination_counts["time_out"] = summary.termination_counts.get(
                "time_out", 0
            ) + int(timeouts.sum().item())

    term_mgr = env.termination_manager
    term_dones = getattr(term_mgr, "_term_dones", None)
    if isinstance(term_dones, torch.Tensor):
        for name in term_mgr.active_terms:
            idx = term_mgr.active_terms.index(name)
            fired = term_dones[:, idx].to(torch.bool)
            if bool(torch.any(fired).item()):
                summary.termination_counts[name] = summary.termination_counts.get(name, 0) + int(
                    fired.sum().item()
                )


def _sim_is_finite(env: ManagerBasedRlEnv) -> bool:
    robot = env.scene["robot"]
    return bool(
        torch.isfinite(robot.data.root_link_pos_w).all().item()
        and torch.isfinite(robot.data.root_link_lin_vel_w).all().item()
        and torch.isfinite(env.sim.data.qpos).all().item()
        and torch.isfinite(env.sim.data.qvel).all().item()
    )


def run_agent_rollout(
    env: ManagerBasedRlEnv,
    action_fn: Callable[[ManagerBasedRlEnv, int], torch.Tensor],
    *,
    num_steps: int,
    record_signatures: bool = False,
    signature_stride: int = 1,
    pre_step_hook: Callable[[ManagerBasedRlEnv, int], None] | None = None,
) -> MdpRolloutSummary:
    """Step the environment with ``action_fn(env, step_idx)`` for ``num_steps``."""
    summary = MdpRolloutSummary()
    ncon_budget = int(env.cfg.sim.nconmax) * env.num_envs
    njmax = int(env.cfg.sim.njmax)
    ep_len = env.episode_length_buf.clone()

    for step_idx in range(num_steps):
        if pre_step_hook is not None:
            pre_step_hook(env, step_idx)
        action = action_fn(env, step_idx)
        _obs, rewards, terminated, truncated, extras = env.step(action)
        summary.steps += 1
        summary.reward_sum += float(rewards.sum().item())

        reward_mgr = env.reward_manager
        dt = env.step_dt
        scale_dt = getattr(env.cfg, "scale_rewards_by_dt", True)
        for term_idx, name in enumerate(reward_mgr.active_terms):
            if name.startswith("_"):
                continue
            rate = float(reward_mgr._step_reward[:, term_idx].sum().item())
            contrib = rate * dt if scale_dt else rate
            summary.reward_term_sums[name] = summary.reward_term_sums.get(name, 0.0) + contrib

        reward_state = ClimbRewardState.get(env)
        summary.max_valid_attachments = max(
            summary.max_valid_attachments,
            int(reward_state.valid_higher_attachment_count.max().item()),
        )
        summary.max_invalid_latch = max(
            summary.max_invalid_latch,
            int(reward_state.invalid_latch_count.max().item()),
        )

        dones = terminated.to(torch.bool) | truncated.to(torch.bool)
        if bool(dones.any().item()):
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            for env_idx in done_ids.tolist():
                if isinstance(env_idx, int):
                    summary.episode_lengths.append(int(ep_len[env_idx].item()) + 1)
            summary.completed_episodes += int(dones.sum().item())
        summary.episode_resets += int(dones.sum().item())

        if not observations_finite(env):
            summary.nonfinite_obs_steps += 1
        if not _sim_is_finite(env):
            summary.nonfinite_sim_steps += 1

        cap = simulation_capacity_snapshot(env)
        summary.max_nacon = max(summary.max_nacon, cap["nacon_total"])
        summary.max_nefc = max(summary.max_nefc, cap["max_nefc"])
        summary.max_nacon_per_env_upper = max(
            summary.max_nacon_per_env_upper, cap["nacon_per_env_upper"]
        )
        summary.max_equalities_per_env = max(
            summary.max_equalities_per_env, cap["max_equalities_per_env"]
        )
        summary.min_ncon_headroom = min(summary.min_ncon_headroom, cap["ncon_headroom"])
        summary.min_njmax_headroom = min(summary.min_njmax_headroom, cap["njmax_headroom"])
        summary.max_active_equalities = max(summary.max_active_equalities, cap["active_equalities"])
        if cap["nacon_total"] >= ncon_budget or cap["max_nefc"] >= njmax:
            raise RuntimeError(
                f"Simulation capacity overflow at step {step_idx}: "
                f"nacon_total={cap['nacon_total']}/{ncon_budget}, "
                f"max_nefc={cap['max_nefc']}/{njmax}"
            )

        _record_terminations(
            env,
            summary,
            terminated=terminated,
            truncated=truncated,
            extras=extras if isinstance(extras, dict) else None,
        )
        if record_signatures and (step_idx % signature_stride == 0):
            summary.signatures.append(rollout_signature(env))
        ep_len = env.episode_length_buf.clone()

    return summary


def stress_random_resets(
    env: ManagerBasedRlEnv,
    *,
    num_resets: int,
    steps_after_reset: int = 3,
    seed: int = 0,
    randomized_ladder: bool = True,
) -> dict[str, int | bool | float]:
    """Exercise randomized resets with post-reset simulation steps.

    When ``randomized_ladder`` is True, each reset genuinely resamples active
    rung count, spacings, ladder pose/tilt, and inactive-rung parking via the
    registered ladder generator.
    """
    stats: dict[str, int | bool | float] = {
        "resets": 0,
        "nonfinite_steps": 0,
        "stale_state_failures": 0,
        "max_nacon": 0,
        "max_nefc": 0,
        "min_ncon_headroom": 1_000_000_000,
        "min_njmax_headroom": 1_000_000_000,
        "ladder_variants": 0,
    }
    ncon_budget = int(env.cfg.sim.nconmax) * env.num_envs
    njmax = int(env.cfg.sim.njmax)
    ladder_snaps: list = []

    for _ in range(num_resets):
        pollute_reset_sensitive_state(env)
        env.reset()
        stats["resets"] = int(stats["resets"]) + 1

        for env_idx in range(env.num_envs):
            try:
                assert_reset_env_clean(env, env_idx=env_idx, ladder_resampled=randomized_ladder)
            except AssertionError:
                stats["stale_state_failures"] = int(stats["stale_state_failures"]) + 1

        ladder_snaps.append(capture_reset_state(env, env_idx=0))

        for step_i in range(steps_after_reset):
            _obs, rewards, _term, _trunc, _extras = env.step(make_zero_action(env))
            if not torch.isfinite(rewards).all():
                stats["nonfinite_steps"] = int(stats["nonfinite_steps"]) + 1
            if not observations_finite(env) or not _sim_is_finite(env):
                stats["nonfinite_steps"] = int(stats["nonfinite_steps"]) + 1
            for env_idx in range(env.num_envs):
                if not history_is_post_reset(env, env_idx=env_idx, steps_since_reset=step_i + 1):
                    stats["stale_state_failures"] = int(stats["stale_state_failures"]) + 1

            cap = simulation_capacity_snapshot(env)
            stats["max_nacon"] = max(int(stats["max_nacon"]), cap["nacon_total"])
            stats["max_nefc"] = max(int(stats["max_nefc"]), cap["max_nefc"])
            stats["min_ncon_headroom"] = min(int(stats["min_ncon_headroom"]), cap["ncon_headroom"])
            stats["min_njmax_headroom"] = min(int(stats["min_njmax_headroom"]), cap["njmax_headroom"])
            if cap["nacon_total"] >= ncon_budget or cap["max_nefc"] >= njmax:
                raise RuntimeError(
                    f"Simulation capacity overflow during reset stress: "
                    f"nacon_total={cap['nacon_total']}/{ncon_budget}, "
                    f"max_nefc={cap['max_nefc']}/{njmax}"
                )

    stats["ladder_variants"] = len(
        {
            (s.ladder_active_count, round(s.ladder_first_height, 4), s.ladder_frame_pos)
            for s in ladder_snaps
        }
    )
    stats["ladder_resampled"] = bool(stats["ladder_variants"] > 1) if randomized_ladder else True
    stats["ok"] = (
        stats["nonfinite_steps"] == 0
        and stats["stale_state_failures"] == 0
        and (not randomized_ladder or stats["ladder_variants"] > 1)
    )
    return stats


def stress_subset_resets(
    env: ManagerBasedRlEnv,
    *,
    reset_env_ids: torch.Tensor,
    preserved_env_ids: torch.Tensor,
    steps_after_reset: int = 3,
) -> dict[str, int | bool]:
    """Subset reset: cleared envs refresh; preserved envs remain unchanged."""
    before = {int(i): capture_reset_state(env, env_idx=int(i)) for i in preserved_env_ids.tolist()}
    pollute_reset_sensitive_state(env, env_ids=reset_env_ids)
    env.reset(env_ids=reset_env_ids)

    stats = {"stale_state_failures": 0, "nonfinite_steps": 0}
    for env_idx in reset_env_ids.tolist():
        try:
            assert_reset_env_clean(env, env_idx=int(env_idx), ladder_resampled=False)
        except AssertionError:
            stats["stale_state_failures"] += 1

    after_reset = {int(i): capture_reset_state(env, env_idx=int(i)) for i in preserved_env_ids.tolist()}
    for env_idx, snap_before in before.items():
        try:
            assert_carryover_preserved(snap_before, after_reset[env_idx])
        except AssertionError:
            stats["stale_state_failures"] += 1

    for step_i in range(steps_after_reset):
        env.step(make_zero_action(env))
        if not observations_finite(env) or not _sim_is_finite(env):
            stats["nonfinite_steps"] += 1
        for env_idx in reset_env_ids.tolist():
            if not history_is_post_reset(env, env_idx=int(env_idx), steps_since_reset=step_i + 1):
                stats["stale_state_failures"] += 1

    stats["ok"] = stats["stale_state_failures"] == 0 and stats["nonfinite_steps"] == 0
    return stats


@dataclass(frozen=True)
class MdpScriptPhase:
    name: str
    steps: int
    kind: ScriptPhaseKind = "neutral"
    latch_left: float = 0.0
    latch_right: float = 0.0


DEFAULT_HOLD_SCRIPT: tuple[MdpScriptPhase, ...] = (
    MdpScriptPhase(name="hold_pose", steps=40, kind="hold"),
)

DEFAULT_FULL_MDP_SCRIPT: tuple[MdpScriptPhase, ...] = (
    MdpScriptPhase(name="hold_pose", steps=40, kind="hold"),
    MdpScriptPhase(name="detach_left", steps=8, kind="detach", latch_left=-1.0),
    MdpScriptPhase(name="invalid_attach", steps=6, kind="invalid_attach", latch_left=1.0),
    MdpScriptPhase(name="reattach_left", steps=10, kind="attach", latch_left=1.0),
    MdpScriptPhase(name="trigger_fall", steps=24, kind="trigger_fall"),
)

# Backward-compatible alias for Stage 6 hold regression on the hold-pose env.
DEFAULT_MDP_SCRIPT = DEFAULT_FULL_MDP_SCRIPT


class _ScriptedPhaseRunner:
    """Shared phased latch/hold stepping without test-side fault injection."""

    def __init__(
        self,
        script: Sequence[MdpScriptPhase],
    ) -> None:
        self.script = tuple(script)
        self._phase_idx = 0
        self._phase_step = 0

    def reset_phase(self) -> None:
        self._phase_idx = 0
        self._phase_step = 0

    @property
    def current_phase(self) -> MdpScriptPhase:
        return self.script[self._phase_idx]

    def advance(self) -> None:
        self._phase_step += 1
        if self._phase_step >= self.current_phase.steps:
            self._phase_step = 0
            if self._phase_idx < len(self.script) - 1:
                self._phase_idx += 1

    def latch_for_phase(self) -> tuple[float, float]:
        phase = self.current_phase
        return phase.latch_left, phase.latch_right


class AssistedHoldScriptedAgent:
    """Stage 6 regression agent: hold-pose env, stiff PD hold only (no root nudge)."""

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        hold_cfg: HoldPoseConfig | None = None,
        script: Sequence[MdpScriptPhase] = DEFAULT_HOLD_SCRIPT,
    ) -> None:
        self.env = env
        self.hold_cfg = hold_cfg or HoldPoseConfig(hold_duration_s=2.0)
        self._runner = _ScriptedPhaseRunner(script)
        self._solution: HoldPoseSolution | None = None
        self._initialized = False

    @property
    def script(self) -> tuple[MdpScriptPhase, ...]:
        return self._runner.script

    def reset(self) -> None:
        self._solution = initialize_hold_pose_scene(self.env, self.hold_cfg)
        self._runner.reset_phase()
        self._initialized = True

    def __call__(self, _env: ManagerBasedRlEnv, _step: int) -> torch.Tensor:
        if not self._initialized or self._solution is None:
            self.reset()
        assert self._solution is not None
        latch_left, latch_right = self._runner.latch_for_phase()
        action = build_hold_action(
            self.env,
            self._solution,
            latch_left=latch_left,
            latch_right=latch_right,
        )
        self._runner.advance()
        return action


class FullMdpScriptedAgent:
    """End-to-end MDP scripted agent using registered managers (latch actions only).

    Assisted hold initialization runs once at ``reset()``. Root nudges for invalid
    attach / fall must be injected by the test harness via ``pre_step_hook`` —
    see ``make_mdp_fault_injection_hook``.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        hold_cfg: HoldPoseConfig | None = None,
        script: Sequence[MdpScriptPhase] = DEFAULT_FULL_MDP_SCRIPT,
    ) -> None:
        self.env = env
        self.hold_cfg = hold_cfg or HoldPoseConfig(hold_duration_s=2.0)
        self._runner = _ScriptedPhaseRunner(script)
        self._initialized = False

    @property
    def script(self) -> tuple[MdpScriptPhase, ...]:
        return self._runner.script

    def reset(self) -> None:
        initialize_hold_pose_scene(self.env, self.hold_cfg)
        self._runner.reset_phase()
        self._initialized = True

    def __call__(self, _env: ManagerBasedRlEnv, _step: int) -> torch.Tensor:
        if not self._initialized:
            self.reset()
        latch_left, latch_right = self._runner.latch_for_phase()
        action = build_latch_action(
            self.env,
            latch_left=latch_left,
            latch_right=latch_right,
        )
        self._runner.advance()
        return action


def make_mdp_fault_injection_hook(
    agent: FullMdpScriptedAgent,
    *,
    invalid_attach_step_m: float = 0.04,
    fall_step_m: float = 0.06,
) -> Callable[[ManagerBasedRlEnv, int], None]:
    """Test-side root nudge during invalid-attach and fall script phases."""

    def _hook(env: ManagerBasedRlEnv, _step: int) -> None:
        kind = agent._runner.current_phase.kind
        if kind == "invalid_attach":
            nudge_robot_away_from_ladder(env, step_m=invalid_attach_step_m)
        elif kind == "trigger_fall":
            nudge_robot_away_from_ladder(env, step_m=fall_step_m)

    return _hook


# Deprecated name kept for imports that expect a single scripted agent class.
ScriptedMdpAgent = AssistedHoldScriptedAgent


def seed_hand_latch(
    env: ManagerBasedRlEnv,
    *,
    hand: str,
    rung_id: int,
    site_id: int = 1,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Force-activate one hand latch for selected environments (test helper)."""
    from train_mimic.tasks.climbing.config.robot import HAND_INDEX
    from train_mimic.tasks.climbing.debug_hold_pose import _seed_hand_latch

    if env_ids is None and env.num_envs == 1:
        _seed_hand_latch(env, hand=hand, rung_id=rung_id, site_id=site_id)
        return

    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    hand_idx = HAND_INDEX[hand]
    eq_id = int(latch.topology.eq_ids[hand_idx, rung_id, site_id].item())
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
    latch.backend.set_active(env_ids, hand_idx, eq_id, active=True)
    latch.state.attached[env_ids, hand_idx] = True
    latch.state.rung_id[env_ids, hand_idx] = rung_id
    latch.state.site_id[env_ids, hand_idx] = site_id
    latch.state.active_eq_id[env_ids, hand_idx] = eq_id
    latch.state.attach_event[env_ids, hand_idx] = False
    latch.state.invalid_attach_request[env_ids, hand_idx] = False
    env.sim.forward()


def run_scripted_hold_regression(
    env: ManagerBasedRlEnv,
    hold_cfg: HoldPoseConfig,
) -> tuple[dict[str, float | bool | str], str]:
    """Headless hold segment used to verify Stage 6 behaviour inside Stage 7."""
    _solution, _metrics, summary, failure = run_headless_hold(env, hold_cfg)
    return summary, failure


def print_rollout_summary(summary: MdpRolloutSummary, *, header: str = "[mdp rollout]") -> None:
    print(header)
    print(f"  steps={summary.steps}")
    print(f"  completed_episodes={summary.completed_episodes}")
    if summary.episode_lengths:
        lengths = summary.episode_lengths
        print(
            f"  episode_length mean={sum(lengths)/len(lengths):.1f} "
            f"min={min(lengths)} max={max(lengths)}"
        )
    print(f"  reward_sum={summary.reward_sum:.4f}")
    if summary.reward_term_sums:
        terms = ", ".join(f"{k}={v:.4f}" for k, v in sorted(summary.reward_term_sums.items()))
        print(f"  reward_terms: {terms}")
    print(f"  max_valid_attachments={summary.max_valid_attachments}")
    print(f"  max_invalid_latch={summary.max_invalid_latch}")
    print(f"  nonfinite_obs_steps={summary.nonfinite_obs_steps}")
    print(f"  nonfinite_sim_steps={summary.nonfinite_sim_steps}")
    print(f"  max_nacon={summary.max_nacon}")
    print(f"  max_nefc={summary.max_nefc}")
    print(f"  max_nacon_per_env_upper={summary.max_nacon_per_env_upper}")
    print(f"  max_equalities_per_env={summary.max_equalities_per_env}")
    print(f"  min_ncon_headroom={summary.min_ncon_headroom}")
    print(f"  min_njmax_headroom={summary.min_njmax_headroom}")
    print(f"  max_active_equalities={summary.max_active_equalities}")
    if summary.termination_counts:
        terms = ", ".join(f"{k}={v}" for k, v in sorted(summary.termination_counts.items()))
        print(f"  terminations: {terms}")


def print_reset_stress_stats(stats: dict[str, int | bool | float], *, header: str = "[reset stress]") -> None:
    print(header)
    for key in (
        "resets",
        "nonfinite_steps",
        "stale_state_failures",
        "ladder_variants",
        "ladder_resampled",
        "max_nacon",
        "max_nefc",
        "min_ncon_headroom",
        "min_njmax_headroom",
        "ok",
    ):
        if key in stats:
            print(f"  {key}={stats[key]}")


def collect_mdp_benchmark_report(
    *,
    seed: int = 42,
    num_envs: int = 1,
    rollout_steps: int = 120,
    num_resets: int = 50,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run zero/random/scripted rollouts and reset stress; return observed summaries."""
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import (
        make_climbing_hold_pose_env_cfg,
        make_climbing_mdp_env_cfg,
    )

    report: dict[str, Any] = {"seed": seed, "device": device}

    pinned = pinned_ladder_cfg()
    mdp_cfg = make_climbing_mdp_env_cfg(
        num_envs=num_envs,
        seed=seed,
        ladder_cfg=pinned,
        episode_length_s=2.0,
        play=True,
    )
    env = ManagerBasedRlEnv(cfg=mdp_cfg, device=device)
    env.reset()

    zero_summary = run_agent_rollout(
        env,
        lambda _env, _step: make_zero_action(_env),
        num_steps=rollout_steps,
    )
    report["zero"] = _summary_to_dict(zero_summary)

    env.reset()
    gen = torch.Generator(device=env.device)
    gen.manual_seed(seed)
    random_summary = run_agent_rollout(
        env,
        lambda _env, _step: make_random_action(_env, generator=gen),
        num_steps=rollout_steps,
    )
    report["random"] = _summary_to_dict(random_summary)

    env.reset()
    mdp_agent = FullMdpScriptedAgent(env)
    fault_hook = make_mdp_fault_injection_hook(mdp_agent)
    scripted_summary = run_agent_rollout(
        env,
        lambda e, s: mdp_agent(e, s),
        num_steps=sum(p.steps for p in mdp_agent.script),
        pre_step_hook=fault_hook,
    )
    report["scripted_mdp"] = _summary_to_dict(scripted_summary)
    env.close()

    hold_cfg = HoldPoseConfig(hold_duration_s=2.0)
    hold_env = ManagerBasedRlEnv(
        cfg=make_climbing_hold_pose_env_cfg(num_envs=1, seed=seed, hold_cfg=hold_cfg),
        device=device,
    )
    hold_env.reset()
    hold_summary, hold_failure = run_scripted_hold_regression(hold_env, hold_cfg)
    report["scripted_hold"] = {"failure": hold_failure, **{k: float(v) if isinstance(v, (int, float)) else v for k, v in hold_summary.items()}}
    hold_env.close()

    rand_cfg = make_climbing_mdp_env_cfg(
        num_envs=1,
        seed=seed,
        ladder_cfg=randomized_ladder_cfg(),
        episode_length_s=1.0,
        play=True,
    )
    stress_env = ManagerBasedRlEnv(cfg=rand_cfg, device=device)
    stress_env.reset()
    report["reset_stress_randomized"] = stress_random_resets(
        stress_env,
        num_resets=num_resets,
        steps_after_reset=3,
        randomized_ladder=True,
    )
    stress_env.close()

    repro_a = _pinned_repro_signatures(seed=seed, device=device)
    repro_b = _pinned_repro_signatures(seed=seed, device=device)
    report["reproducibility_match"] = repro_a == repro_b
    report["repro_signatures_len"] = len(repro_a)

    return report


def _summary_to_dict(summary: MdpRolloutSummary) -> dict[str, Any]:
    lengths = summary.episode_lengths
    return {
        "steps": summary.steps,
        "completed_episodes": summary.completed_episodes,
        "episode_length_mean": sum(lengths) / len(lengths) if lengths else None,
        "episode_length_min": min(lengths) if lengths else None,
        "episode_length_max": max(lengths) if lengths else None,
        "reward_sum": summary.reward_sum,
        "reward_term_sums": dict(summary.reward_term_sums),
        "max_valid_attachments": summary.max_valid_attachments,
        "max_invalid_latch": summary.max_invalid_latch,
        "termination_counts": dict(summary.termination_counts),
        "max_nacon": summary.max_nacon,
        "max_nefc": summary.max_nefc,
        "max_equalities_per_env": summary.max_equalities_per_env,
        "min_ncon_headroom": summary.min_ncon_headroom,
        "min_njmax_headroom": summary.min_njmax_headroom,
        "nonfinite_obs_steps": summary.nonfinite_obs_steps,
        "nonfinite_sim_steps": summary.nonfinite_sim_steps,
    }


def _pinned_repro_signatures(*, seed: int, device: str) -> list[tuple[float, ...]]:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

    from train_mimic.tasks.climbing.config.env import make_climbing_mdp_env_cfg

    env = ManagerBasedRlEnv(
        cfg=make_climbing_mdp_env_cfg(num_envs=1, seed=seed, ladder_cfg=pinned_ladder_cfg()),
        device=device,
    )
    env.reset()
    gen = torch.Generator(device=env.device)
    gen.manual_seed(99)
    actions = [0, 1, 0, -1, 1]
    sigs: list[tuple[float, ...]] = []
    for step_idx in range(len(actions)):
        action = make_zero_action(env) if actions[step_idx] == 0 else make_random_action(env, generator=gen)
        env.step(action)
        sigs.append(rollout_signature(env))
    env.close()
    return sigs
