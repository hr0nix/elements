import cmath
import math

import numpy as np
import pytest

from elements.quantum.operator import Operator
from elements.quantum.spin import Spin, TwoSpins
from elements.quantum.state import StateSpace

sigma_x, sigma_y, sigma_z = Spin.sigma_x, Spin.sigma_y, Spin.sigma_z
identity = Spin.identity
sigma_dot_tau = (
    sigma_x.tensor_product(sigma_x)
    + sigma_y.tensor_product(sigma_y)
    + sigma_z.tensor_product(sigma_z)
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


def test_a_mixture_is_a_density_whose_eigenvalues_are_the_probabilities():
    mixture = Operator.spectral([(0.3, Spin.z_plus), (0.7, Spin.z_minus)])
    assert mixture.is_density
    assert sigma_z.expectation(mixture) == pytest.approx(0.3 * 1 + 0.7 * -1)
    assert sigma_x.expectation(mixture) == pytest.approx(0)
    with pytest.raises(ValueError, match="density"):
        sigma_z.expectation(sigma_x)


def test_a_pure_density_gives_the_expectations_of_its_ket():
    pure = Operator.density(Spin.y_plus)
    for sigma in (sigma_x, sigma_y, sigma_z):
        assert sigma.expectation(pure) == pytest.approx(sigma.expectation(Spin.y_plus))


def test_densities_come_from_normalized_kets_and_only_densities_have_a_purity():
    with pytest.raises(ValueError, match="normalized ket"):
        Operator.density(2 * Spin.z_plus)
    with pytest.raises(ValueError, match="densities"):
        sigma_x.purity()


def test_tensor_product_builds_operators_on_a_pair_of_spins():
    ud, du = TwoSpins.ud, TwoSpins.du
    sigma_z_on_first = sigma_z.tensor_product(identity)
    sigma_z_on_second = identity.tensor_product(sigma_z)
    flip_both = sigma_x.tensor_product(sigma_x)
    assert (sigma_z_on_first @ ud).close_to(ud)
    assert (sigma_z_on_second @ ud).close_to(-ud)
    assert not (sigma_z_on_first @ ud).close_to(sigma_z_on_second @ ud)
    assert (flip_both @ ud).close_to(du)


def test_partial_trace_of_a_product_scales_the_kept_factor_by_the_trace_of_the_rest():
    a, b = sigma_x + 2 * identity, sigma_y + 3 * identity
    product = a.tensor_product(b)
    assert product.partial_trace(keep=0).close_to(b.trace() * a)
    assert product.partial_trace(keep=1).close_to(a.trace() * b)


def test_partial_trace_keeps_several_factors_in_their_original_order():
    a, b, c = sigma_x + 2 * identity, sigma_y + 3 * identity, sigma_z + 4 * identity
    product = a.tensor_product(b, c)
    assert product.partial_trace(keep=1).close_to(a.trace() * c.trace() * b)
    assert product.partial_trace(keep=[0, 2]).close_to(b.trace() * a.tensor_product(c))
    assert product.partial_trace(keep=[2, 0]).close_to(b.trace() * a.tensor_product(c))
    assert product.partial_trace(keep=[0, 1, 2]).close_to(product)


def test_each_half_of_a_singlet_is_maximally_mixed():
    singlet = TwoSpins.singlet @ TwoSpins.singlet.dagger
    for kept in (0, 1):
        assert singlet.partial_trace(keep=kept).close_to(identity / 2)


def test_partial_trace_rejects_unknown_factors_and_keeping_nothing():
    two_spins = identity.tensor_product(identity)
    with pytest.raises(ValueError, match="between 0 and 1"):
        two_spins.partial_trace(keep=2)
    with pytest.raises(ValueError, match="at least one factor"):
        two_spins.partial_trace(keep=[])


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
    eigenpairs = list(zip(operator.eigenvalues, operator.eigenbasis, strict=True))
    assert len(eigenpairs) == operator.state_space.dimension
    for value, vector in eigenpairs:
        assert (operator @ vector).close_to(vector * value)
    assert Operator.spectral(eigenpairs).close_to(operator)


def test_hermitian_operators_list_their_eigenpairs_by_increasing_eigenvalue():
    assert list(sigma_z.eigenvalues) == [-1, 1]
    assert sigma_z.eigenbasis[0].close_to(Spin.z_minus)
    assert sigma_dot_tau.eigenvalues == pytest.approx([-3, 1, 1, 1])


def test_real_symmetric_operators_get_real_eigenvectors():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(6, 6))
    operator = Operator(StateSpace([str(label) for label in range(6)]), values + values.T)
    for value, vector in zip(operator.eigenvalues, operator.eigenbasis, strict=True):
        assert not vector.components.imag.any()
        assert (operator @ vector).close_to(vector * value)


def test_functions_of_an_operator_act_on_its_eigenvalues():
    hamiltonian = sigma_x + 2 * sigma_z
    assert hamiltonian.map_eigenvalues(np.square).close_to(hamiltonian @ hamiltonian)
    inverse = hamiltonian.map_eigenvalues(lambda energy: 1 / energy)
    assert (hamiltonian @ inverse).close_to(identity)


def test_non_normal_operators_have_no_eigenbasis_to_exponentiate():
    shear = identity + Spin.z_plus @ Spin.z_minus.dagger
    with pytest.raises(ValueError, match="normal"):
        _ = shear.eigenbasis
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
    assert (-1j * theta * sigma).exp().close_to(closed_form)


def test_imaginary_exponentials_of_a_hermitian_operator_are_unitaries_whose_exponents_add():
    hamiltonian = sigma_x + 2 * sigma_z
    evolution = (-0.3j * hamiltonian).exp()
    assert (evolution.dagger @ evolution).close_to(identity)
    assert ((-0.1j * hamiltonian).exp() @ (-0.2j * hamiltonian).exp()).close_to(evolution)


def test_spin_in_a_magnetic_field_along_z_precesses_around_it():
    omega, t = 2.0, 0.4
    hamiltonian = omega / 2 * sigma_z
    state = (-1j * t * hamiltonian).exp() @ Spin.x_plus
    assert sigma_x.expectation(state) == pytest.approx(math.cos(omega * t))
    assert sigma_y.expectation(state) == pytest.approx(math.sin(omega * t))
    assert sigma_z.expectation(state) == pytest.approx(0)


def test_energy_eigenstates_only_pick_up_a_phase():
    t = 0.9
    evolution = (-1j * t * sigma_dot_tau).exp()
    assert (evolution @ TwoSpins.singlet).close_to(cmath.exp(3j * t) * TwoSpins.singlet)
    for triplet in (TwoSpins.triplet_1, TwoSpins.triplet_2, TwoSpins.triplet_3):
        assert (evolution @ triplet).close_to(cmath.exp(-1j * t) * triplet)
