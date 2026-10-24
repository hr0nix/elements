import pytest
import math
import numpy as np

from elements.quantum.spin import Spin, TwoSpins
from elements.quantum.evolution import EigenbasisEvolution

sigma_x, sigma_y, sigma_z = Spin.sigma_x, Spin.sigma_y, Spin.sigma_z
sigma_x_tau_x = sigma_x.tensor_product(sigma_x)
sigma_y_tau_y = sigma_y.tensor_product(sigma_y)
sigma_z_tau_z = sigma_z.tensor_product(sigma_z)


def test_exercise_6_6():
    sigma_x_tau_y = sigma_x.tensor_product(sigma_y)
    assert sigma_x_tau_y.expectation(TwoSpins.singlet) == pytest.approx(0)


def test_exercise_6_7():
    t1 = TwoSpins.triplet_1
    assert sigma_x_tau_x.expectation(t1) == pytest.approx(1)
    assert sigma_y_tau_y.expectation(t1) == pytest.approx(1)
    assert sigma_z_tau_z.expectation(t1) == pytest.approx(-1)


def test_exercise_6_8():
    t2, t3 = TwoSpins.triplet_2, TwoSpins.triplet_3
    assert sigma_x_tau_x.expectation(t2) == pytest.approx(1)
    assert sigma_y_tau_y.expectation(t2) == pytest.approx(-1)
    assert sigma_z_tau_z.expectation(t2) == pytest.approx(1)
    assert sigma_x_tau_x.expectation(t3) == pytest.approx(-1)
    assert sigma_y_tau_y.expectation(t3) == pytest.approx(1)
    assert sigma_z_tau_z.expectation(t3) == pytest.approx(1)


def test_exercise_6_9():
    sigma_dot_tau = sigma_x_tau_x + sigma_y_tau_y + sigma_z_tau_z
    assert (sigma_dot_tau @ TwoSpins.singlet).close_to(-3 * TwoSpins.singlet)
    for triplet in (TwoSpins.triplet_1, TwoSpins.triplet_2, TwoSpins.triplet_3):
        assert (sigma_dot_tau @ triplet).close_to(triplet)


def test_exercise_6_10():
    w = 0.7
    H = 0.5 * w * (sigma_x_tau_x + sigma_y_tau_y + sigma_z_tau_z)
    energies = H.eigenvalues
    assert energies == pytest.approx([-1.5 * w, 0.5 * w, 0.5 * w, 0.5 * w])

    # UU only picks up a phase
    from_uu = EigenbasisEvolution(H, TwoSpins.uu)
    assert from_uu.state_at(1.0).close_to(
        np.exp(-1j * w * 0.5 * 1.0) * TwoSpins.uu)
    assert from_uu.state_at(1.3).normalized.close_to(
        np.exp(-1j * w * 0.5 * 1.3) * TwoSpins.uu)

    # UD picks up a phase and a relative phase
    assert EigenbasisEvolution(H, TwoSpins.ud).state_at(1.6).normalized.close_to(
        np.sqrt(1/2) * np.exp(-1j * w * 0.5 * 1.6) * (
            TwoSpins.triplet_1 +
            np.exp(1j * w * 2.0 * 1.6) * TwoSpins.singlet
        )
    )