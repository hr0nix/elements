import numpy as np
import pytest

from elements.quantum.basis import Basis
from elements.quantum.spin import Spin

z_basis = Spin.state_space.basis
x_basis = Basis([Spin.x_plus, Spin.x_minus])
y_basis = Basis([Spin.y_plus, Spin.y_minus])
state = 0.6 * Spin.z_plus + 0.8j * Spin.z_minus


def test_coordinates_are_the_amplitudes_along_the_basis_kets():
    assert x_basis.coordinates(Spin.z_plus) == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert x_basis.coordinates(Spin.x_minus) == pytest.approx([0, 1])


def test_a_ket_is_rebuilt_from_its_coordinates_in_any_basis():
    for basis in (z_basis, x_basis, y_basis):
        assert basis.ket(basis.coordinates(state)).close_to(state)


def test_overlaps_between_two_bases_change_coordinates_from_one_to_the_other():
    assert x_basis.overlaps(y_basis) @ y_basis.coordinates(state) == pytest.approx(
        x_basis.coordinates(state)
    )


def test_a_basis_must_be_orthonormal_and_span_its_space():
    with pytest.raises(ValueError, match="orthonormal"):
        Basis([Spin.z_plus, Spin.x_plus])
    with pytest.raises(ValueError, match="dimensions"):
        Basis([Spin.z_plus])


def test_orthonormal_kets_are_completed_to_a_basis_by_kets_orthogonal_to_them():
    basis = Basis.completing([Spin.x_plus])
    assert basis[0].close_to(Spin.x_plus)
    assert abs(basis[1].dagger @ Spin.x_minus) == pytest.approx(1)
