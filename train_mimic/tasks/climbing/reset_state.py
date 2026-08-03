"""Reset-state capture and verification for Stage 7 MDP robustness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from train_mimic.tasks.climbing.config.observations import PROPRIO_HISTORY_LENGTH
from train_mimic.tasks.climbing.ladder.contacts import ClimbContactState
from train_mimic.tasks.climbing.ladder.latch import ClimbLatchState
from train_mimic.tasks.climbing.ladder.reward_state import ClimbRewardState
from train_mimic.tasks.climbing.ladder.state import LadderRuntime
from train_mimic.tasks.climbing.mdp.observations import action_term_slice

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class ResetStateSnapshot:
    """Compact per-environment MDP state for reset regression checks."""

    env_idx: int
    eq_active_count: int
    latch_attached: tuple[bool, bool]
    latch_rung_id: tuple[int, int]
    latch_site_id: tuple[int, int]
    latch_active_eq_id: tuple[int, int]
    latch_invalid_attach: tuple[bool, bool]
    contact_hand_in_contact: tuple[bool, bool]
    contact_hand_rung_id: tuple[int, int]
    reward_max_attachment_h: float
    reward_valid_higher_attachments: int
    reward_invalid_latch_count: int
    reward_success_rewarded: bool
    prev_joint_action: tuple[float, ...]
    prev_latch_action: tuple[float, float]
    history_num_pushes: int
    history_prev_latch_t0: tuple[float, float]
    episode_length: int
    ladder_active_count: int
    ladder_first_height: float
    ladder_frame_pos: tuple[float, float, float]
    metrics_max_head: float | None = None


# mjlab observation history receives one push when reset computes initial obs.
HISTORY_PUSHES_AFTER_RESET = 1


def history_baseline_after_reset() -> int:
    return HISTORY_PUSHES_AFTER_RESET


def _history_num_pushes(env: ManagerBasedRlEnv, *, env_idx: int) -> int:
    obs_mgr = env.observation_manager
    buffers = getattr(obs_mgr, "_group_obs_term_history_buffer", {})
    group = buffers.get("actor_proprio_history", {})
    term_buf = group.get("prev_latch_action")
    if term_buf is None:
        return 0
    pushes = getattr(term_buf, "_num_pushes", None)
    if pushes is None:
        return 0
    return int(pushes[env_idx].item())


def _history_prev_latch_t0(env: ManagerBasedRlEnv, *, env_idx: int) -> tuple[float, float]:
    obs_mgr = env.observation_manager
    buffers = getattr(obs_mgr, "_group_obs_term_history_buffer", {})
    group = buffers.get("actor_proprio_history", {})
    term_buf = group.get("prev_latch_action")
    if term_buf is None:
        return (0.0, 0.0)
    buf = getattr(term_buf, "_buffer", None)
    if buf is None:
        return (0.0, 0.0)
    # CircularBuffer stores [time, batch, dim]; index 0 is most recent after pushes.
    values = buf[0, env_idx].detach().cpu().tolist()
    return (float(values[0]), float(values[1]))


def capture_reset_state(env: ManagerBasedRlEnv, *, env_idx: int = 0) -> ResetStateSnapshot:
    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    contacts = ClimbContactState.get(env)
    contacts.update()
    reward = ClimbRewardState.get(env)
    runtime = LadderRuntime.get(env)
    sample = runtime.sample
    assert sample is not None

    eq_active = int(env.sim.data.eq_active[env_idx].to(torch.int64).sum().item())
    prev_joint = action_term_slice(env.action_manager.action, env, "joint_pos")[env_idx]
    prev_latch = action_term_slice(env.action_manager.action, env, "latch")[env_idx]

    metrics_max_head: float | None = None
    metrics_mgr = getattr(env, "metrics_manager", None)
    if metrics_mgr is not None and "max_head_height_l" in metrics_mgr.active_terms:
        idx = metrics_mgr.active_terms.index("max_head_height_l")
        metrics_max_head = float(metrics_mgr._step_values[env_idx, idx].item())

    frame_pos = sample.frame_pos[env_idx].detach().cpu().tolist()
    active = sample.active_mask[env_idx]
    active_heights = sample.rung_heights[env_idx][active]
    first_h = float(active_heights[0].item()) if bool(active.any().item()) else float("nan")

    return ResetStateSnapshot(
        env_idx=env_idx,
        eq_active_count=eq_active,
        latch_attached=tuple(bool(v) for v in latch.state.attached[env_idx].tolist()),
        latch_rung_id=tuple(int(v) for v in latch.state.rung_id[env_idx].tolist()),
        latch_site_id=tuple(int(v) for v in latch.state.site_id[env_idx].tolist()),
        latch_active_eq_id=tuple(int(v) for v in latch.state.active_eq_id[env_idx].tolist()),
        latch_invalid_attach=tuple(bool(v) for v in latch.state.invalid_attach_request[env_idx].tolist()),
        contact_hand_in_contact=tuple(bool(v) for v in contacts.state.hand_in_contact[env_idx].tolist()),
        contact_hand_rung_id=tuple(int(v) for v in contacts.state.hand_rung_id[env_idx].tolist()),
        reward_max_attachment_h=float(reward.max_rewarded_attachment_height_l[env_idx].item()),
        reward_valid_higher_attachments=int(reward.valid_higher_attachment_count[env_idx].item()),
        reward_invalid_latch_count=int(reward.invalid_latch_count[env_idx].item()),
        reward_success_rewarded=bool(reward.success_rewarded[env_idx].item()),
        prev_joint_action=tuple(float(v) for v in prev_joint.detach().cpu().tolist()),
        prev_latch_action=(float(prev_latch[0].item()), float(prev_latch[1].item())),
        history_num_pushes=_history_num_pushes(env, env_idx=env_idx),
        history_prev_latch_t0=_history_prev_latch_t0(env, env_idx=env_idx),
        episode_length=int(env.episode_length_buf[env_idx].item()),
        ladder_active_count=int(sample.num_active[env_idx].item()),
        ladder_first_height=first_h,
        ladder_frame_pos=(float(frame_pos[0]), float(frame_pos[1]), float(frame_pos[2])),
        metrics_max_head=metrics_max_head,
    )


def pollute_reset_sensitive_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
) -> None:
    """Write obviously stale values into every reset-sensitive store."""
    if env_ids is None:
        env_ids = slice(None)

    latch = ClimbLatchState.get(env)
    latch._ensure_state()
    latch.state.attached[env_ids] = True
    latch.state.rung_id[env_ids] = 2
    latch.state.site_id[env_ids] = 1
    latch.state.active_eq_id[env_ids] = 0
    latch.state.invalid_attach_request[env_ids] = True
    latch.state.latch_command[env_ids] = 1.0

    contacts = ClimbContactState.get(env)
    if contacts.state is not None:
        contacts.state.hand_in_contact[env_ids] = True
        contacts.state.hand_rung_id[env_ids] = 2

    reward = ClimbRewardState.get(env)
    reward.max_rewarded_attachment_height_l[env_ids] = 1.5
    reward.max_rewarded_progress_height_l[env_ids] = 9.0
    reward.valid_higher_attachment_count[env_ids] = 7
    reward.invalid_latch_count[env_ids] = 5
    reward.success_rewarded[env_ids] = True
    reward.time_to_success_steps[env_ids] = 123

    # Distinctive previous actions so stale action/history buffers are detectable.
    action = env.action_manager.action.clone()
    action_term_slice(action, env, "joint_pos")[env_ids] = 3.14
    action_term_slice(action, env, "latch")[env_ids] = 0.99
    env.action_manager.action[:] = action


def assert_reset_env_clean(
    env: ManagerBasedRlEnv,
    *,
    env_idx: int,
    ladder_resampled: bool,
) -> None:
    """Assert one environment has post-reset defaults across all tracked stores."""
    snap = capture_reset_state(env, env_idx=env_idx)
    assert snap.eq_active_count == 0, f"env {env_idx}: stale eq_active={snap.eq_active_count}"
    assert not any(snap.latch_attached), f"env {env_idx}: stale latch attached={snap.latch_attached}"
    assert all(r == -1 for r in snap.latch_rung_id), snap.latch_rung_id
    assert all(s == -1 for s in snap.latch_site_id), snap.latch_site_id
    assert all(e == -1 for e in snap.latch_active_eq_id), snap.latch_active_eq_id
    assert not any(snap.latch_invalid_attach), snap.latch_invalid_attach
    assert snap.reward_max_attachment_h == float("-inf"), snap.reward_max_attachment_h
    assert snap.reward_valid_higher_attachments == 0
    assert snap.reward_invalid_latch_count == 0
    assert not snap.reward_success_rewarded
    assert abs(snap.history_prev_latch_t0[0]) < 0.5 and abs(snap.history_prev_latch_t0[1]) < 0.5, (
        f"env {env_idx}: stale history latch={snap.history_prev_latch_t0}"
    )
    assert snap.episode_length == 0, f"env {env_idx}: episode_length={snap.episode_length}"
    if ladder_resampled:
        # Only a weak check: active count must be within configured bounds.
        assert 4 <= snap.ladder_active_count <= 10


def assert_carryover_preserved(
    before: ResetStateSnapshot,
    after: ResetStateSnapshot,
) -> None:
    """Subset-reset: ladder/latch/reward carry-over unchanged on preserved envs."""
    assert before.eq_active_count == after.eq_active_count
    assert before.latch_attached == after.latch_attached
    assert before.latch_rung_id == after.latch_rung_id
    assert before.latch_site_id == after.latch_site_id
    assert before.latch_active_eq_id == after.latch_active_eq_id
    assert before.reward_max_attachment_h == after.reward_max_attachment_h
    assert before.reward_valid_higher_attachments == after.reward_valid_higher_attachments
    assert before.reward_invalid_latch_count == after.reward_invalid_latch_count
    assert before.reward_success_rewarded == after.reward_success_rewarded
    assert before.ladder_active_count == after.ladder_active_count
    assert before.ladder_first_height == after.ladder_first_height
    assert before.ladder_frame_pos == after.ladder_frame_pos


def assert_env_unchanged(
    before: ResetStateSnapshot,
    after: ResetStateSnapshot,
) -> None:
    """Full snapshot equality (pinned reproducibility checks)."""
    assert before == after


def ladder_resampled_across_resets(samples: list[ResetStateSnapshot]) -> bool:
    """Return True when ladder fingerprints differ across resets."""
    if len(samples) < 2:
        return False
    keys = {
        (s.ladder_active_count, round(s.ladder_first_height, 4), s.ladder_frame_pos)
        for s in samples
    }
    return len(keys) > 1


def history_is_post_reset(
    env: ManagerBasedRlEnv,
    *,
    env_idx: int,
    steps_since_reset: int,
    reset_baseline: int | None = None,
) -> bool:
    """History buffer push count tracks post-reset obs + stepped frames."""
    pushes = _history_num_pushes(env, env_idx=env_idx)
    baseline = history_baseline_after_reset() if reset_baseline is None else reset_baseline
    expected = min(baseline + steps_since_reset, PROPRIO_HISTORY_LENGTH)
    return pushes == expected
