"""Tests for advsteer.data.steering_direction — the pluggable steering-vector
readout used by the attack (mean_diff) and the hyperplane experiment.

All pure-logic: synthetic tensors, no model download.

Run:  uv run pytest tests/test_steering_direction.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from advsteer.data import steering_direction


# ---------------------------------------------------------------------------
# mean_diff (must reproduce the pre-existing readout exactly)
# ---------------------------------------------------------------------------

def test_mean_diff_matches_definition():
    torch.manual_seed(0)
    h_pos = torch.randn(20, 8)
    h_neg = torch.randn(20, 8)
    got = steering_direction(h_pos, h_neg, "mean_diff")
    assert torch.equal(got, h_pos.mean(0) - h_neg.mean(0))


def test_mean_diff_is_the_default():
    torch.manual_seed(0)
    h_pos, h_neg = torch.randn(5, 4), torch.randn(5, 4)
    assert torch.equal(steering_direction(h_pos, h_neg),
                       steering_direction(h_pos, h_neg, "mean_diff"))


# ---------------------------------------------------------------------------
# hyperplane (RepE PCA reading direction)
# ---------------------------------------------------------------------------

def test_hyperplane_is_unit_norm_and_sign_oriented():
    torch.manual_seed(1)
    h_neg = torch.randn(30, 12)
    w = torch.randn(12)
    w = w / w.norm()
    # POS = NEG shifted by strictly positive per-pair multiples of w.
    coeffs = torch.rand(30) + 1.0  # in [1, 2]
    h_pos = h_neg + coeffs.unsqueeze(1) * w
    d = steering_direction(h_pos, h_neg, "hyperplane")

    assert d.shape == (12,)
    assert math.isclose(d.norm().item(), 1.0, rel_tol=1e-5)
    # Sign invariant: POS projects at least as high as NEG onto the direction.
    assert (h_pos @ d).mean().item() >= (h_neg @ d).mean().item()
    # Dominant contrast direction is w, so the readout should align with it.
    assert torch.dot(d, w).item() > 0.99


def test_hyperplane_recovers_top_variance_direction():
    torch.manual_seed(2)
    d_model = 16
    u = torch.randn(d_model)
    u = u / u.norm()
    # Paired differences vary predominantly along u (dominant principal comp).
    coeffs = torch.linspace(-2, 2, 40)  # centered → sign ~arbitrary, use |cos|
    h_neg = torch.randn(40, d_model)
    h_pos = h_neg + coeffs.unsqueeze(1) * u + 0.001 * torch.randn(40, d_model)
    d = steering_direction(h_pos, h_neg, "hyperplane")
    assert abs(torch.dot(d, u).item()) > 0.99


def test_hyperplane_is_the_max_variance_direction_of_diffs():
    # By definition the top PC maximizes the variance of the (centered) paired
    # differences among all unit vectors — verify against random unit vectors.
    torch.manual_seed(3)
    h_pos, h_neg = torch.randn(25, 10), torch.randn(25, 10)
    d = steering_direction(h_pos, h_neg, "hyperplane")
    diffs = h_pos - h_neg
    diffs = diffs - diffs.mean(0, keepdim=True)
    var_d = ((diffs @ d) ** 2).sum().item()
    for _ in range(20):
        r = torch.randn(10)
        r = r / r.norm()
        assert var_d + 1e-4 >= ((diffs @ r) ** 2).sum().item()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_unknown_method_raises():
    h = torch.randn(3, 4)
    with pytest.raises(ValueError):
        steering_direction(h, h, "svm")


def test_shape_guards():
    with pytest.raises(AssertionError):  # mismatched N
        steering_direction(torch.randn(3, 4), torch.randn(4, 4), "hyperplane")
    with pytest.raises(AssertionError):  # not 2-D
        steering_direction(torch.randn(4), torch.randn(4), "mean_diff")
