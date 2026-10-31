from elements.quantum.spin import Spin
from elements.quantum.state import StateSpace, StateSpaceVector
from elements.quantum.operator import Operator
from elements.quantum.gates import Gates
import numpy as np
import pytest
from typing import Sequence


deflection_space = StateSpace("+", "-")
deflection_plus = deflection_space.ket("+")
deflection_minus = deflection_space.ket("-")
# The deflection register starts out in "-", and the magnet flips it to "+" for the spin it
# deflects that way.
deflection_initial = deflection_space.ket("-")


def make_magnet_operator(positive_deflection_on: StateSpaceVector) -> Operator:
    swap = Gates.flip(deflection_space)
    return Gates.controlled(
        positive_deflection_on, swap
    )


def marginal_probabilities(state: StateSpaceVector, register: int | Sequence[int] = 1) -> np.ndarray:
    return Operator.density(state).partial_trace(keep=register).probabilities()


def test_stern_gerlach_split_up_down_twice():
    def _apply_circuit(input: StateSpaceVector) -> StateSpaceVector:
        x = input
        
        x = x.tensor_product(deflection_initial)
        
        magnet1 = make_magnet_operator(positive_deflection_on=Spin.up)
        x = magnet1 @ x

        x = x.tensor_product(deflection_initial)
        
        magnet2_u = make_magnet_operator(positive_deflection_on=Spin.up.tensor_product(deflection_plus))
        x = magnet2_u @ x
        magnet2_d = make_magnet_operator(positive_deflection_on=Spin.up.tensor_product(deflection_minus))
        x = magnet2_d @ x

        return x

    x = _apply_circuit(Spin.up)
    assert marginal_probabilities(x, register=1) == pytest.approx([1.0, 0.0])
    assert marginal_probabilities(x, register=[1,2]) == pytest.approx(np.array([[1.0, 0.0], [0.0, 0.0]]))

    x = _apply_circuit(Spin.x_plus)
    assert marginal_probabilities(x, register=1) == pytest.approx([0.5, 0.5])
    assert marginal_probabilities(x, register=[1,2]) == pytest.approx(np.array([[0.5, 0.0], [0.0, 0.5]]))


def test_stern_gerlach_split_up_down_then_left_right():
    def _apply_circuit(input: StateSpaceVector) -> StateSpaceVector:
        x = input
        
        x = x.tensor_product(deflection_initial)
        
        magnet1 = make_magnet_operator(positive_deflection_on=Spin.up)
        x = magnet1 @ x

        x = x.tensor_product(deflection_initial)
        
        magnet2_u = make_magnet_operator(positive_deflection_on=Spin.x_plus.tensor_product(deflection_plus))
        x = magnet2_u @ x
        magnet2_d = make_magnet_operator(positive_deflection_on=Spin.x_plus.tensor_product(deflection_minus))
        x = magnet2_d @ x

        return x

    x = _apply_circuit(Spin.up)
    assert marginal_probabilities(x, register=1) == pytest.approx([1.0, 0.0])
    assert marginal_probabilities(x, register=[1,2]) == pytest.approx(np.array([[0.5, 0.5], [0.0, 0.0]]))

    x = _apply_circuit(Spin.x_plus)
    assert marginal_probabilities(x, register=1) == pytest.approx([0.5, 0.5])
    assert marginal_probabilities(x, register=[1,2]) == pytest.approx(np.array([[0.25, 0.25], [0.25, 0.25]]))


def test_stern_gerlach_split_up_down_then_left_right_then_up_down_again():
    def _apply_circuit(input: StateSpaceVector) -> StateSpaceVector:
        x = input
        
        x = x.tensor_product(deflection_initial)
        
        magnet1 = make_magnet_operator(positive_deflection_on=Spin.up)
        x = magnet1 @ x

        x = x.tensor_product(deflection_initial)
        
        magnet2_u = make_magnet_operator(positive_deflection_on=Spin.x_plus.tensor_product(deflection_plus))
        x = magnet2_u @ x
        magnet2_d = make_magnet_operator(positive_deflection_on=Spin.x_plus.tensor_product(deflection_minus))
        x = magnet2_d @ x

        x = x.tensor_product(deflection_initial)
        magnet3_ul = make_magnet_operator(positive_deflection_on=Spin.up.tensor_product(deflection_plus).tensor_product(deflection_plus))
        x = magnet3_ul @ x
        magnet3_ur = make_magnet_operator(positive_deflection_on=Spin.up.tensor_product(deflection_plus).tensor_product(deflection_minus))
        x = magnet3_ur @ x
        magnet3_dl = make_magnet_operator(positive_deflection_on=Spin.up.tensor_product(deflection_minus).tensor_product(deflection_plus))
        x = magnet3_dl @ x
        magnet3_dr = make_magnet_operator(positive_deflection_on=Spin.up.tensor_product(deflection_minus).tensor_product(deflection_minus))
        x = magnet3_dr @ x

        return x

    x = _apply_circuit(Spin.up)
    assert marginal_probabilities(x, register=1) == pytest.approx([1.0, 0.0])
    assert marginal_probabilities(x, register=[1,2]) == pytest.approx(np.array([[0.5, 0.5], [0.0, 0.0]]))
    assert marginal_probabilities(x, register=[1,2,3]) == pytest.approx(np.array([[[0.25, 0.25], [0.25, 0.25]], [[0.0, 0.0], [0.0, 0.0]]]))

    x = _apply_circuit(Spin.x_plus)
    assert marginal_probabilities(x, register=1) == pytest.approx([0.5, 0.5])
    assert marginal_probabilities(x, register=[1,2]) == pytest.approx(np.array([[0.25, 0.25], [0.25, 0.25]]))
    assert marginal_probabilities(x, register=[1,2,3]) == pytest.approx(np.array([[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]]))

