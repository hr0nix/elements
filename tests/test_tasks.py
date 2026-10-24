import numpy as np
import pytest

from elements.quantum.continuous_approx import (
    CyclicalGrid1DStateSpace,
    Operators1D,
    WaveFunctions1D,
)
from elements.quantum.evolution import EigenbasisEvolution
from elements.quantum.tasks import FreeParticle, Gravity, HarmonicOscillator, Task

state = CyclicalGrid1DStateSpace(-30.0, 30.0, 601)
x, p = Operators1D.position(state), Operators1D.momentum(state)


def evolution(task: Task, x0: float, p0: float, sigma: float, mass: float) -> EigenbasisEvolution:
    packet = WaveFunctions1D.packet(state, x0, p0, sigma)
    return EigenbasisEvolution(task.hamiltonian(state, mass), packet)


def spread(packet) -> float:
    return np.sqrt((x @ x).expectation(packet) - x.expectation(packet) ** 2)


def assert_follows_classical(task: Task, x0: float, p0: float, mass: float, times) -> None:
    packets = evolution(task, x0, p0, sigma=1.0, mass=mass)
    for time in times:
        packet = packets.state_at(time)
        classical_x, classical_p = task.classical(x0, p0, mass, time)
        assert x.expectation(packet) == pytest.approx(classical_x, abs=1e-6)
        assert p.expectation(packet) == pytest.approx(classical_p, abs=1e-6)


def test_free_packet_expectations_follow_the_classical_particle():
    assert_follows_classical(FreeParticle(), x0=-5.0, p0=2.0, mass=1.5, times=(0.0, 1.0, 4.0))


def test_free_packet_spreads_as_the_textbook_formula_says():
    sigma, mass = 0.8, 1.0
    packets = evolution(FreeParticle(), x0=0.0, p0=1.0, sigma=sigma, mass=mass)
    for time in (0.0, 2.0, 5.0):
        expected = sigma * np.sqrt(1 + (time / (2 * mass * sigma**2)) ** 2)
        assert spread(packets.state_at(time)) == pytest.approx(expected)


def test_free_packet_momentum_distribution_does_not_change():
    packets = evolution(FreeParticle(), x0=-3.0, p0=1.5, sigma=1.0, mass=0.7)
    _, initial = WaveFunctions1D.momentum_wavefunction(packets.state_at(0.0))
    _, later = WaveFunctions1D.momentum_wavefunction(packets.state_at(6.0))
    assert np.allclose(np.abs(initial), np.abs(later))


def test_oscillator_levels_are_spaced_by_omega():
    task, mass = HarmonicOscillator(), 2.0
    omega = np.sqrt(task.spring_constant / mass)
    energies = np.sort(task.hamiltonian(state, mass).eigenvalues.real)[:5]
    assert energies == pytest.approx(omega * (np.arange(5) + 0.5))


def test_oscillator_expectations_follow_the_classical_orbit_exactly():
    times = (0.0, 3.0, 7.5, 20.0)
    assert_follows_classical(HarmonicOscillator(), x0=6.0, p0=1.0, mass=1.5, times=times)


def test_oscillator_returns_a_classical_particle_to_its_start_after_one_period():
    task, mass = HarmonicOscillator(), 1.5
    period = 2 * np.pi / np.sqrt(task.spring_constant / mass)
    assert task.classical(6.0, 1.0, mass, period) == pytest.approx((6.0, 1.0))


def test_coherent_state_keeps_its_width_while_other_packets_breathe():
    task, mass = HarmonicOscillator(), 1.0
    coherent_width = np.sqrt(1 / (2 * mass * np.sqrt(task.spring_constant / mass)))
    coherent = evolution(task, x0=-8.0, p0=0.0, sigma=coherent_width, mass=mass)
    squeezed = evolution(task, x0=-8.0, p0=0.0, sigma=0.5 * coherent_width, mass=mass)
    for time in (1.0, 2.5, 4.0):
        assert spread(coherent.state_at(time)) == pytest.approx(coherent_width)
        assert spread(squeezed.state_at(time)) != pytest.approx(0.5 * coherent_width, rel=0.05)


def test_gravity_levels_are_airy_zeros():
    task, mass = Gravity(g=0.5), 1.0
    scale = (mass * task.g**2 / 2) ** (1 / 3)
    first_zero_of_airy_derivative, first_zero_of_airy = 1.018792971647471, 2.338107410459767
    energies = np.sort(task.hamiltonian(state, mass).eigenvalues.real)[:2]
    expected = scale * np.array([first_zero_of_airy_derivative, first_zero_of_airy])
    assert energies == pytest.approx(expected, rel=1e-3)


def test_falling_packet_follows_the_classical_parabola_until_it_reaches_the_object():
    assert_follows_classical(Gravity(g=0.5), x0=-15.0, p0=0.0, mass=1.0, times=(0.0, 2.0, 4.0))


def test_classical_particle_falls_through_the_object_and_conserves_energy():
    task, mass, x0, p0 = Gravity(g=0.5), 2.0, -6.0, 1.0
    energy = p0**2 / (2 * mass) + mass * task.g * abs(x0)
    positions = []
    for time in np.linspace(0.0, 40.0, 81):
        position, momentum = task.classical(x0, p0, mass, time)
        positions.append(position)
        assert momentum**2 / (2 * mass) + mass * task.g * abs(position) == pytest.approx(energy)
    assert min(positions) < 0 < max(positions)
    assert task.classical(x0, p0, mass, 0.0) == pytest.approx((x0, p0))
