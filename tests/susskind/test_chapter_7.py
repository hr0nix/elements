import math

import numpy as np
import pytest

from elements.quantum.operator import Operator
from elements.quantum.spin import Spin, TwoSpins

density = Operator.density


def test_projection_operator_properties():
    some_state = Spin.x_plus
    orthogonal_state = Spin.x_minus
    proj = some_state @ some_state.dagger
    assert proj.dagger.close_to(proj)
    assert (proj @ some_state).close_to(some_state)
    assert (proj @ orthogonal_state).close_to(0.0 * orthogonal_state)
    assert (proj @ proj).close_to(proj)
    assert proj.trace() == pytest.approx(1)
    assert (proj + orthogonal_state @ orthogonal_state.dagger).close_to(Spin.identity)


def test_density_matrix_form():
    assert density(Spin.z_plus).matrix == pytest.approx(np.array([[1.0, 0.0], [0.0, 0.0]]))
    assert density(Spin.z_minus).matrix == pytest.approx(np.array([[0.0, 0.0], [0.0, 1.0]]))
    assert density(Spin.x_plus).matrix == pytest.approx(np.array([[0.5, 0.5], [0.5, 0.5]]))
    superposition = np.sqrt(0.3) * Spin.z_plus + np.sqrt(0.7) * Spin.z_minus
    assert density(superposition).matrix == pytest.approx(
        np.array([[0.3, np.sqrt(0.3 * 0.7)], [np.sqrt(0.7 * 0.3), 0.7]])
    )

    assert density(Spin.z_plus).purity() == pytest.approx(1.0)
    assert density(Spin.z_minus).purity() == pytest.approx(1.0)
    assert density(Spin.x_plus).purity() == pytest.approx(1.0)
    mixture = 0.3 * density(Spin.z_plus) + 0.7 * density(Spin.z_minus)
    assert mixture.purity() == pytest.approx(0.3 ** 2 + 0.7 ** 2)


def test_expectation_via_trace():
    some_operator = Spin.sigma_z
    some_state = math.sqrt(0.25) * Spin.z_plus + math.sqrt(0.75) * Spin.z_minus
    expectation = some_operator.expectation(some_state)
    assert expectation == pytest.approx(0.25 * 1.0 + 0.75 * (-1.0))
    assert (some_state @ some_state.dagger @ some_operator).trace() == pytest.approx(expectation)


def test_example_7_6():
    state = np.sqrt(1/2) * (TwoSpins.ud + TwoSpins.du)
    d = density(state)
    assert d.purity() == pytest.approx(1.0)

    d_first = d.partial_trace(keep=0)
    assert d_first.purity() == pytest.approx(0.5)
    assert d_first.matrix == pytest.approx(np.array([[0.5, 0.0], [0.0, 0.5]]))

    d_second = d.partial_trace(keep=1)
    assert d_second.purity() == pytest.approx(0.5)
    assert d_second.matrix == pytest.approx(np.array([[0.5, 0.0], [0.0, 0.5]]))


def test_exercide_7_8():
    # Clearly non-entangled state
    psi_1 = 0.5 * (TwoSpins.uu + TwoSpins.ud + TwoSpins.du + TwoSpins.dd)
    d_1 = density(psi_1)
    assert d_1.purity() == pytest.approx(1.0)
    assert d_1.partial_trace(keep=0).purity() == pytest.approx(1.0)
    assert d_1.partial_trace(keep=1).purity() == pytest.approx(1.0)

    # Entangled state
    psi_2 = np.sqrt(0.5) * (TwoSpins.uu + TwoSpins.dd)
    d_2 = density(psi_2)
    assert d_2.purity() == pytest.approx(1.0)
    assert d_2.partial_trace(keep=0).purity() == pytest.approx(0.5)
    assert d_2.partial_trace(keep=1).purity() == pytest.approx(0.5)

    # Non-entangled state, can be factorized
    psi_3 = 0.2 * (3.0 * TwoSpins.uu + 4.0 * TwoSpins.ud)
    d_3 = density(psi_3)
    assert d_3.purity() == pytest.approx(1.0)
    assert d_3.partial_trace(keep=0).purity() == pytest.approx(1.0)
    assert d_3.partial_trace(keep=1).purity() == pytest.approx(1.0)
