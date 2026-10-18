import itertools

import numpy as np
import pytest

from elements.quantum.spin import Spin, TwoSpins

AXES = {
    "z": (Spin.z_plus, Spin.z_minus),
    "x": (Spin.x_plus, Spin.x_minus),
    "y": (Spin.y_plus, Spin.y_minus),
}


@pytest.mark.parametrize("axis", AXES)
def test_up_and_down_along_each_axis_form_an_orthonormal_basis(axis):
    up, down = AXES[axis]
    assert up.is_normalized
    assert down.is_normalized
    assert up.dagger @ down == pytest.approx(0)


def test_spin_prepared_along_one_axis_is_a_coin_flip_along_another():
    for first_axis, second_axis in itertools.combinations(AXES.values(), 2):
        for prepared, measured in itertools.product(first_axis, second_axis):
            probability = abs(measured.dagger @ prepared) ** 2
            assert probability == pytest.approx(0.5)


def test_inner_product_of_product_states_factorizes():
    a, b, c, d = Spin.x_plus, Spin.y_minus, Spin.z_plus, Spin.x_plus
    combined = a.kronecker_product(b).dagger @ c.kronecker_product(d)
    separate = (a.dagger @ c) * (b.dagger @ d)
    assert combined == pytest.approx(separate)


def test_singlet_and_triplets_form_an_orthonormal_basis():
    basis = [TwoSpins.singlet, TwoSpins.triplet_1, TwoSpins.triplet_2, TwoSpins.triplet_3]
    overlaps = [[a.dagger @ b for b in basis] for a in basis]
    assert np.allclose(overlaps, np.eye(4))
