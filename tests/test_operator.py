import numpy as np
import pytest

from elements.quantum.operator import Operator
from elements.quantum.spin import Spin, TwoSpins

sigma_x, sigma_y, sigma_z = Spin.sigma_x, Spin.sigma_y, Spin.sigma_z
identity = Spin.identity


def test_pauli_matrices_built_from_their_eigenstates_match_the_textbook():
    assert np.allclose(sigma_x.matrix, [[0, 1], [1, 0]])
    assert np.allclose(sigma_y.matrix, [[0, -1j], [1j, 0]])
    assert np.allclose(sigma_z.matrix, [[1, 0], [0, -1]])


def test_operators_can_be_assembled_from_outer_products():
    up, down = Spin.z_plus, Spin.z_minus
    assert (up @ up.dagger - down @ down.dagger).close_to(sigma_z)
    assert (up @ down.dagger + down @ up.dagger).close_to(sigma_x)
    assert (-1j * (up @ down.dagger) + 1j * (down @ up.dagger)).close_to(sigma_y)


def test_pauli_matrices_square_to_identity_and_multiply_cyclically():
    for sigma in (sigma_x, sigma_y, sigma_z):
        assert (sigma @ sigma).close_to(identity)
    assert (sigma_x @ sigma_y).close_to(1j * sigma_z)
    assert (sigma_y @ sigma_z).close_to(1j * sigma_x)
    assert (sigma_z @ sigma_x).close_to(1j * sigma_y)
    assert not (sigma_x @ sigma_y).close_to(sigma_y @ sigma_x)


def test_spin_expectation_values_are_one_along_the_prepared_axis_and_zero_across_it():
    x_plus, z_plus = Spin.x_plus, Spin.z_plus
    assert sigma_x.expectation(x_plus) == pytest.approx(1)
    assert sigma_x.expectation(z_plus) == pytest.approx(0)


def test_expectation_requires_a_hermitian_operator():
    with pytest.raises(ValueError, match="Hermitian"):
        (sigma_x @ sigma_y).expectation(Spin.z_plus)


def test_expectation_requires_a_normalized_ket():
    with pytest.raises(ValueError, match="normalized ket"):
        sigma_z.expectation(2 * Spin.z_plus)
    with pytest.raises(ValueError, match="normalized ket"):
        sigma_z.expectation(Spin.z_plus.dagger)


def test_kronecker_product_builds_operators_on_a_pair_of_spins():
    ud, du = TwoSpins.ud, TwoSpins.du
    sigma_z_on_first = sigma_z.kronecker_product(identity)
    sigma_z_on_second = identity.kronecker_product(sigma_z)
    flip_both = sigma_x.kronecker_product(sigma_x)
    assert (sigma_z_on_first @ ud).close_to(ud)
    assert (sigma_z_on_second @ ud).close_to(-ud)
    assert not (sigma_z_on_first @ ud).close_to(sigma_z_on_second @ ud)
    assert (flip_both @ ud).close_to(du)


@pytest.mark.parametrize(
    ("operator", "hermitian", "unitary"),
    [
        pytest.param(sigma_x, True, True, id="pauli"),
        pytest.param(sigma_x + sigma_z, True, False, id="sum of paulis"),
        pytest.param(sigma_x @ sigma_y, False, True, id="product of paulis"),
        pytest.param(Operator.spectral([(1, Spin.x_plus)]), True, False, id="projector"),
        pytest.param(
            Operator.spectral([(1, Spin.z_plus), (1j, Spin.z_minus)]), False, True, id="phase"
        ),
    ],
)
def test_hermiticity_and_unitarity_are_read_off_the_spectrum(operator, hermitian, unitary):
    assert operator.is_hermitian == hermitian
    assert operator.is_unitary == unitary


def test_eigenvalues_alone_cannot_tell_a_shear_from_the_identity():
    raising = Spin.z_plus @ Spin.z_minus.dagger
    shear = identity + raising
    assert np.allclose(shear.eigenvalues, identity.eigenvalues)
    assert not shear.is_normal
    assert not shear.is_hermitian
    assert not shear.is_unitary


def test_spectral_requires_orthonormal_eigenvectors():
    with pytest.raises(ValueError, match="orthonormal"):
        Operator.spectral([(+1, Spin.z_plus), (-1, Spin.x_plus)])
