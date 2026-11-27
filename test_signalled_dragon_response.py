"""Tests for the probe-only confidence-weighted dragon-signal response."""

from __future__ import annotations

import numpy as np
import pytest

from probe_signalled_dragon_response import confidence_weighted_probs


def test_confidence_mixes_policy_with_its_pass_free_distribution():
    baseline = np.asarray([0.60, 0.30, 0.10], dtype=np.float64)

    assert np.allclose(confidence_weighted_probs(baseline, 0, 0.0), baseline)
    half_confident = confidence_weighted_probs(baseline, 0, 0.5)
    assert np.allclose(half_confident, [0.30, 0.525, 0.175])
    assert half_confident[0] == pytest.approx((1.0 - 0.5) * baseline[0])
    assert half_confident[1] / half_confident[2] == pytest.approx(3.0)
    assert np.allclose(
        confidence_weighted_probs(baseline, 0, 1.0), [0.0, 0.75, 0.25]
    )


def test_confidence_keeps_pass_if_there_is_no_other_policy_action():
    only_pass = np.asarray([1.0, 0.0], dtype=np.float64)
    assert np.allclose(confidence_weighted_probs(only_pass, 0, 1.0), only_pass)


def test_confidence_must_be_a_probability():
    with pytest.raises(ValueError, match="between 0 and 1"):
        confidence_weighted_probs(np.asarray([0.5, 0.5]), 0, 1.01)
