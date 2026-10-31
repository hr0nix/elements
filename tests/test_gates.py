import numpy as np
import pytest

from elements.quantum.gates import Gates, ensure_unitary
from elements.quantum.operator import Operator
from elements.quantum.spin import Spin
from elements.quantum.state import StateSpace

path = StateSpace("upper", "lower")
upper, lower = path.ket("upper"), path.ket("lower")


def test_flip_swaps_the_two_basis_kets():
    flip = Gates.flip(path)
    assert np.allclose(flip.matrix, [[0, 1], [1, 0]])
    assert (flip @ upper).close_to(lower)
    assert (flip @ lower).close_to(upper)


def test_a_flip_controlled_by_spin_down_is_the_textbook_cnot():
    cnot = Gates.controlled(Spin.down, Gates.flip(path))
    assert np.allclose(cnot.matrix, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


def test_a_controlled_gate_acts_on_the_target_only_when_the_control_is_in_the_given_state():
    controlled = Gates.controlled(Spin.x_minus, Gates.flip(path))
    assert (controlled @ Spin.x_plus.tensor_product(upper)).close_to(
        Spin.x_plus.tensor_product(upper)
    )
    assert (controlled @ Spin.x_minus.tensor_product(upper)).close_to(
        Spin.x_minus.tensor_product(lower)
    )


def test_a_controlled_gate_entangles_a_control_in_superposition_with_the_target():
    controlled = Gates.controlled(Spin.x_minus, Gates.flip(path))
    entangled = (Spin.x_plus.tensor_product(upper) + Spin.x_minus.tensor_product(lower)) / np.sqrt(2)
    assert (controlled @ Spin.z_plus.tensor_product(upper)).close_to(entangled)


def test_a_controlled_gate_needs_a_unitary_target():
    with pytest.raises(ValueError, match="controlled must return a unitary operator"):
        Gates.controlled(Spin.up, Operator.density(upper))


def test_ensure_unitary_rejects_operators_that_are_not_unitary():
    @ensure_unitary
    def projector() -> Operator:
        return Operator.density(Spin.z_plus)

    with pytest.raises(ValueError, match="projector must return a unitary operator"):
        projector()
