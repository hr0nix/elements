import itertools

import numpy as np
import pytest

from elements.quantum.spin import Spin, TwoSpins
from elements.quantum.state import StateSpace

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
    combined = a.tensor_product(b).dagger @ c.tensor_product(d)
    separate = (a.dagger @ c) * (b.dagger @ d)
    assert combined == pytest.approx(separate)


def test_basis_kets_of_a_product_space_are_products_of_basis_kets():
    spin = Spin.state_space
    for first, second in itertools.product(spin.labels, repeat=2):
        product = spin.ket(first).tensor_product(spin.ket(second))
        assert TwoSpins.state_space.ket(first + second).close_to(product)


def test_tensor_products_remember_their_factors_regardless_of_grouping():
    spin = Spin.state_space
    left = spin.tensor_product(spin).tensor_product(spin)
    right = spin.tensor_product(spin.tensor_product(spin))
    assert spin.factors == (spin,)
    assert left.factors == (spin, spin, spin)
    assert left == right


def test_a_space_with_product_like_labels_is_not_a_product_space():
    flat = StateSpace(*TwoSpins.state_space.labels)
    assert flat.labels == TwoSpins.state_space.labels
    assert flat != TwoSpins.state_space


def test_separately_built_spaces_with_the_same_labels_are_equal():
    assert StateSpace("u", "d") == StateSpace("u", "d")
    assert StateSpace("u", "d") != StateSpace("d", "u")


def test_a_product_space_can_be_given_its_own_labels():
    spin = Spin.state_space
    arrows = StateSpace("↑↑", "↑↓", "↓↑", "↓↓", factors=[spin, spin])
    assert arrows.factors == (spin, spin)
    assert arrows != TwoSpins.state_space
    three_spins = StateSpace(*(str(i) for i in range(8)), factors=[TwoSpins.state_space, spin])
    assert three_spins.factors == (spin, spin, spin)


def test_equal_state_spaces_can_stand_for_each_other_as_dictionary_keys():
    spin = Spin.state_space
    rebuilt = StateSpace("u", "d").tensor_product(StateSpace("u", "d"))
    assert {TwoSpins.state_space: "two spins"}[rebuilt] == "two spins"
    assert hash(StateSpace("u", "d")) == hash(spin)


def test_factor_dimensions_must_multiply_to_the_number_of_labels():
    spin = Spin.state_space
    with pytest.raises(ValueError, match="product of the factor dimensions"):
        StateSpace("a", "b", "c", factors=[spin, spin])


def test_factors_can_be_omitted_but_not_empty():
    assert StateSpace("a").factors == (StateSpace("a"),)
    with pytest.raises(ValueError, match="at least one factor"):
        StateSpace("a", factors=[])


def test_labels_are_written_out_one_by_one():
    with pytest.raises(TypeError, match="unpack"):
        StateSpace(["u", "d"])


def test_singlet_and_triplets_form_an_orthonormal_basis():
    basis = [TwoSpins.singlet, TwoSpins.triplet_1, TwoSpins.triplet_2, TwoSpins.triplet_3]
    overlaps = [[a.dagger @ b for b in basis] for a in basis]
    assert np.allclose(overlaps, np.eye(4))


def test_printing_a_state_skips_components_that_are_close_to_zero():
    noisy = TwoSpins.ud + 1e-12 * TwoSpins.du
    assert repr(noisy) == "1.00|ud>"
    assert repr(0 * TwoSpins.ud) == "0"


def test_printing_a_state_writes_signs_between_terms_and_drops_zero_parts():
    assert repr(TwoSpins.singlet) == "0.71|ud> - 0.71|du>"
    assert repr(-Spin.y_minus) == "-0.71|u> + 0.71j|d>"
    assert repr(Spin.x_plus.dagger) == "0.71<u| + 0.71<d|"
    assert repr((0.6 - 0.8j) * Spin.z_plus) == "(0.60-0.80j)|u>"
