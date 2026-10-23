import cmath
import math

import numpy as np
import pytest

from elements.quantum.operator import Operator
from elements.quantum.spin import Spin, TwoSpins

sigma_x, sigma_y, sigma_z = Spin.sigma_x, Spin.sigma_y, Spin.sigma_z
identity = Spin.identity
sigma_dot_tau = (
    sigma_x.kronecker_product(sigma_x)
    + sigma_y.kronecker_product(sigma_y)
    + sigma_z.kronecker_product(sigma_z)
)


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


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(sigma_x + sigma_z, id="hermitian"),
        pytest.param(sigma_x @ sigma_y, id="unitary"),
        pytest.param(sigma_dot_tau, id="degenerate hermitian"),
        pytest.param(1j * sigma_dot_tau, id="degenerate anti-hermitian"),
        pytest.param(Operator.spectral([(1, Spin.x_plus)]), id="projector"),
    ],
)
def test_operator_is_rebuilt_from_its_own_eigenpairs(operator):
    eigenpairs = list(zip(operator.eigenvalues, operator.eigenvectors, strict=True))
    assert len(eigenpairs) == operator.state_space.dimension
    for value, vector in eigenpairs:
        assert (operator @ vector).close_to(vector * value)
    assert Operator.spectral(eigenpairs).close_to(operator)


def test_non_normal_operators_have_no_eigenbasis_to_exponentiate():
    shear = identity + Spin.z_plus @ Spin.z_minus.dagger
    with pytest.raises(ValueError, match="normal"):
        _ = shear.eigenvectors
    with pytest.raises(ValueError, match="normal"):
        shear.exp()


@pytest.mark.parametrize(
    "sigma",
    [
        pytest.param(sigma_x, id="x"),
        pytest.param(sigma_y, id="y"),
        pytest.param(sigma_z, id="z"),
        pytest.param((sigma_x + sigma_z) / math.sqrt(2), id="xz diagonal"),
    ],
)
def test_exponentials_of_operators_squaring_to_identity_have_a_closed_form(sigma):
    theta = 0.7
    closed_form = math.cos(theta) * identity - 1j * math.sin(theta) * sigma
    assert sigma.exp(-1j * theta).close_to(closed_form)


def test_imaginary_exponentials_of_a_hermitian_operator_are_unitaries_whose_exponents_add():
    hamiltonian = sigma_x + 2 * sigma_z
    evolution = hamiltonian.exp(-0.3j)
    assert (evolution.dagger @ evolution).close_to(identity)
    assert (hamiltonian.exp(-0.1j) @ hamiltonian.exp(-0.2j)).close_to(evolution)


def test_spin_in_a_magnetic_field_along_z_precesses_around_it():
    omega, t = 2.0, 0.4
    hamiltonian = omega / 2 * sigma_z
    state = hamiltonian.exp(-1j * t) @ Spin.x_plus
    assert sigma_x.expectation(state) == pytest.approx(math.cos(omega * t))
    assert sigma_y.expectation(state) == pytest.approx(math.sin(omega * t))
    assert sigma_z.expectation(state) == pytest.approx(0)


def test_energy_eigenstates_only_pick_up_a_phase():
    t = 0.9
    evolution = sigma_dot_tau.exp(-1j * t)
    assert (evolution @ TwoSpins.singlet).close_to(cmath.exp(3j * t) * TwoSpins.singlet)
    for triplet in (TwoSpins.triplet_1, TwoSpins.triplet_2, TwoSpins.triplet_3):
        assert (evolution @ triplet).close_to(cmath.exp(-1j * t) * triplet)
