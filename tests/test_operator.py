import numpy as np
import pytest

from elements.quantum.operator import Operator
from elements.quantum.spin import Spin

sigma_x = Operator.spectral([(+1, Spin.x_plus()), (-1, Spin.x_minus())])
sigma_y = Operator.spectral([(+1, Spin.y_plus()), (-1, Spin.y_minus())])
sigma_z = Operator.spectral([(+1, Spin.z_plus()), (-1, Spin.z_minus())])
identity = Operator.identity(Spin.state_space())


def test_pauli_matrices_built_from_their_eigenstates_match_the_textbook():
    assert np.allclose(sigma_x.matrix, [[0, 1], [1, 0]])
    assert np.allclose(sigma_y.matrix, [[0, -1j], [1j, 0]])
    assert np.allclose(sigma_z.matrix, [[1, 0], [0, -1]])


def test_operators_can_be_assembled_from_outer_products():
    up, down = Spin.z_plus(), Spin.z_minus()
    up_bra, down_bra = up.conjugate_transpose(), down.conjugate_transpose()
    assert up @ up_bra - down @ down_bra == sigma_z
    assert up @ down_bra + down @ up_bra == sigma_x
    assert -1j * (up @ down_bra) + 1j * (down @ up_bra) == sigma_y


def test_pauli_matrices_square_to_identity_and_multiply_cyclically():
    for sigma in (sigma_x, sigma_y, sigma_z):
        assert sigma @ sigma == identity
    assert sigma_x @ sigma_y == 1j * sigma_z
    assert sigma_y @ sigma_z == 1j * sigma_x
    assert sigma_z @ sigma_x == 1j * sigma_y
    assert sigma_x @ sigma_y != sigma_y @ sigma_x


def test_eigenstates_have_definite_values_and_other_axes_average_to_zero():
    x_plus, z_plus = Spin.x_plus(), Spin.z_plus()
    assert x_plus.conjugate_transpose() @ sigma_x @ x_plus == pytest.approx(1)
    assert z_plus.conjugate_transpose() @ sigma_x @ z_plus == pytest.approx(0)


@pytest.mark.parametrize(
    ("operator", "hermitian", "unitary"),
    [
        pytest.param(sigma_x, True, True, id="pauli"),
        pytest.param(sigma_x + sigma_z, True, False, id="sum of paulis"),
        pytest.param(sigma_x @ sigma_y, False, True, id="product of paulis"),
        pytest.param(Operator.spectral([(1, Spin.x_plus())]), True, False, id="projector"),
        pytest.param(
            Operator.spectral([(1, Spin.z_plus()), (1j, Spin.z_minus())]), False, True, id="phase"
        ),
    ],
)
def test_hermiticity_and_unitarity_are_read_off_the_spectrum(operator, hermitian, unitary):
    assert operator.is_hermitian == hermitian
    assert operator.is_unitary == unitary


def test_eigenvalues_alone_cannot_tell_a_shear_from_the_identity():
    raising = Spin.z_plus() @ Spin.z_minus().conjugate_transpose()
    shear = identity + raising
    assert np.allclose(shear.eigenvalues, identity.eigenvalues)
    assert not shear.is_normal
    assert not shear.is_hermitian
    assert not shear.is_unitary


def test_spectral_requires_orthonormal_eigenvectors():
    with pytest.raises(ValueError, match="orthonormal"):
        Operator.spectral([(+1, Spin.z_plus()), (-1, Spin.x_plus())])
