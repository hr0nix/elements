import pytest

from elements.quantum.spin import Spin, TwoSpins

sigma_x, sigma_y, sigma_z = Spin.sigma_x, Spin.sigma_y, Spin.sigma_z
sigma_x_tau_x = sigma_x.kronecker_product(sigma_x)
sigma_y_tau_y = sigma_y.kronecker_product(sigma_y)
sigma_z_tau_z = sigma_z.kronecker_product(sigma_z)


def test_exercise_6_6():
    sigma_x_tau_y = sigma_x.kronecker_product(sigma_y)
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
