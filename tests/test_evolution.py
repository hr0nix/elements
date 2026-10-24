import pytest

from elements.quantum.continuous_approx import (
    CyclicalGrid1DStateSpace,
    Operators1D,
    WaveFunctions1D,
)
from elements.quantum.evolution import EigenbasisEvolution
from elements.quantum.spin import Spin, TwoSpins


def test_eigenbasis_evolution_agrees_with_exponentiating_the_hamiltonian():
    hamiltonian = 0.35 * (
        Spin.sigma_x.kronecker_product(Spin.sigma_x)
        + Spin.sigma_y.kronecker_product(Spin.sigma_y)
        + Spin.sigma_z.kronecker_product(Spin.sigma_z)
    )
    evolution = EigenbasisEvolution(hamiltonian, TwoSpins.ud)
    for time in (0.0, 0.7, 2.3):
        expected = hamiltonian.exp(-1j * time) @ TwoSpins.ud
        assert evolution.state_at(time).close_to(expected)


def test_eigenbasis_evolution_keeps_states_normalized():
    state = CyclicalGrid1DStateSpace(-10.0, 10.0, 101)
    hamiltonian = Operators1D.hamiltonian(state, mass=1.0, v=lambda x: 0.5 * x**2)
    packet = WaveFunctions1D.packet(state, x0=2.0, p0=0.5, sigma=1.0)
    assert EigenbasisEvolution(hamiltonian, packet).state_at(7.3).norm == pytest.approx(1.0)


def test_energy_components_hold_the_whole_state_and_give_its_mean_energy():
    state = CyclicalGrid1DStateSpace(-10.0, 10.0, 101)
    hamiltonian = Operators1D.hamiltonian(state, mass=1.0, v=lambda x: 0.5 * x**2)
    packet = WaveFunctions1D.packet(state, x0=2.0, p0=0.5, sigma=1.0)
    evolution = EigenbasisEvolution(hamiltonian, packet)
    weights = abs(evolution.energy_components) ** 2
    assert weights.sum() == pytest.approx(1.0)
    assert (weights * evolution.energies).sum() == pytest.approx(hamiltonian.expectation(packet))


def test_evolution_needs_a_hermitian_hamiltonian():
    state = CyclicalGrid1DStateSpace(-10.0, 10.0, 21)
    packet = WaveFunctions1D.packet(state, x0=2.0, p0=0.5, sigma=1.0)
    with pytest.raises(ValueError, match="Hermitian"):
        EigenbasisEvolution(Operators1D.translation(state, 0.3), packet)


def test_initial_state_must_come_from_the_hamiltonian_state_space():
    with pytest.raises(ValueError, match="state space"):
        EigenbasisEvolution(Spin.sigma_z.kronecker_product(Spin.sigma_z), Spin.up)
