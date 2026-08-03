from __future__ import annotations

import pytest
import torch

from train_mimic.tasks.tracking.rl.runner import (
    _aggregate_ep_extra_values,
    _ep_extras_keys,
    _format_duration,
    _one_based_iteration_range,
    _resolve_total_iterations,
)


def test_one_based_iteration_range_starts_at_one_for_fresh_run() -> None:
    assert list(_one_based_iteration_range(0, 10)) == list(range(1, 11))


def test_one_based_iteration_range_resumes_from_completed_iteration() -> None:
    assert list(_one_based_iteration_range(10, 12)) == [11, 12]


def test_one_based_iteration_range_is_empty_when_already_at_target() -> None:
    assert list(_one_based_iteration_range(10, 10)) == []


def test_one_based_iteration_range_rejects_target_below_completed() -> None:
    with pytest.raises(ValueError, match='num_learning_iterations'):
        _one_based_iteration_range(11, 10)


def test_resolve_total_iterations_preserves_fresh_run_count() -> None:
    assert _resolve_total_iterations(0, 10) == 10


def test_resolve_total_iterations_adds_requested_iterations_on_resume() -> None:
    assert _resolve_total_iterations(10, 12) == 22


def test_resolve_total_iterations_rejects_negative_requested_iterations() -> None:
    with pytest.raises(ValueError, match='non-negative'):
        _resolve_total_iterations(10, -1)


def test_format_duration_keeps_hours_above_one_day() -> None:
    assert _format_duration(33 * 3600 + 16 * 60 + 25) == "33:16:25"


def test_aggregate_ep_extra_sums_termination_counts() -> None:
    values = torch.tensor([5.0, 2.0, 0.0, 7.0])
    total = _aggregate_ep_extra_values("Episode_Termination/fall", values)
    assert float(total.item()) == pytest.approx(14.0)


def test_aggregate_ep_extra_means_non_termination_keys() -> None:
    values = torch.tensor([5.0, 2.0, 0.0, 7.0])
    mean = _aggregate_ep_extra_values("Episode_Metrics/success", values)
    assert float(mean.item()) == pytest.approx(3.5)


def test_ep_extras_keys_unions_across_empty_leading_dicts() -> None:
    keys = _ep_extras_keys(
        [
            {},
            {"Episode_Termination/fall": 2},
            {"Episode_Metrics/success": 0.5, "Episode_Termination/fall": 1},
        ]
    )
    assert keys == ["Episode_Termination/fall", "Episode_Metrics/success"]
