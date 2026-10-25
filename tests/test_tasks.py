import numpy as np
import pytest
from scipy.special import ai_zeros

from elements.quantum.continuous_approx import (
    ContinuousBasis,
    CyclicalGrid1DStateSpace,
    FreeParticle,
    Gravity,
    HarmonicOscillator,
    Operators1D,
    States1D,
    Task1D,
)
from elements.quantum.state import StateSpaceVector

grid = CyclicalGrid1DStateSpace(-30.0, 30.0, 601)
x, p = Operators1D.position(grid), Operators1D.momentum(grid)


def evolve(task: Task1D, x0: float, p0: float, sigma: float, time: float) -> StateSpaceVector:
    packet = States1D.packet(grid, x0, p0, sigma)
    return task.hamiltonian(grid).unitary(time) @ packet


def spread(packet) -> float:
    return np.sqrt((x @ x).expectation(packet) - x.expectation(packet) ** 2)


def assert_follows_classical(task: Task1D, x0: float, p0: float, times) -> None:
    for time in times:
        packet = evolve(task, x0, p0, sigma=1.0, time=time)
        classical_x, classical_p = task.classical_state(x0, p0, time)
        assert x.expectation(packet) == pytest.approx(classical_x, abs=1e-6)
        assert p.expectation(packet) == pytest.approx(classical_p, abs=1e-6)


def test_free_packet_expectations_follow_the_classical_particle():
    assert_follows_classical(FreeParticle(mass=1.5), x0=-5.0, p0=2.0, times=(0.0, 1.0, 4.0))


def test_free_packet_spreads_as_the_textbook_formula_says():
    sigma, mass = 0.8, 1.0
    for time in (0.0, 2.0, 5.0):
        packet = evolve(FreeParticle(mass=mass), x0=0.0, p0=1.0, sigma=sigma, time=time)
        expected = sigma * np.sqrt(1 + (time / (2 * mass * sigma**2)) ** 2)
        assert spread(packet) == pytest.approx(expected)


def test_free_packet_momentum_distribution_does_not_change():
    task, momentum = FreeParticle(mass=0.7), ContinuousBasis(p)
    initial = momentum.wavefunction(evolve(task, x0=-3.0, p0=1.5, sigma=1.0, time=0.0))
    later = momentum.wavefunction(evolve(task, x0=-3.0, p0=1.5, sigma=1.0, time=6.0))
    assert np.allclose(np.abs(initial), np.abs(later))


def test_oscillator_levels_are_spaced_by_omega():
    task = HarmonicOscillator(mass=2.0)
    energies = task.hamiltonian(grid).eigenvalues[:5]
    assert energies == pytest.approx(task.omega * (np.arange(5) + 0.5))


def test_oscillator_expectations_follow_the_classical_orbit_exactly():
    times = (0.0, 3.0, 7.5, 20.0)
    assert_follows_classical(HarmonicOscillator(mass=1.5), x0=6.0, p0=1.0, times=times)


def test_oscillator_returns_a_classical_particle_to_its_start_after_one_period():
    task = HarmonicOscillator(mass=1.5)
    period = 2 * np.pi / task.omega
    assert task.classical_state(6.0, 1.0, period) == pytest.approx((6.0, 1.0))


def test_coherent_state_keeps_its_width_while_other_packets_breathe():
    task = HarmonicOscillator(mass=1.0)
    coherent_width = np.sqrt(1 / (2 * task.mass * task.omega))
    for time in (1.0, 2.5, 4.0):
        coherent = evolve(task, x0=-8.0, p0=0.0, sigma=coherent_width, time=time)
        squeezed = evolve(task, x0=-8.0, p0=0.0, sigma=0.5 * coherent_width, time=time)
        assert spread(coherent) == pytest.approx(coherent_width)
        assert spread(squeezed) != pytest.approx(0.5 * coherent_width, rel=0.05)


def test_gravity_levels_are_airy_zeros():
    task = Gravity(mass=1.0, g=0.5)
    scale = (task.mass * task.g**2 / 2) ** (1 / 3)
    # Levels even about x = 0 have zero slope there, so they sit at zeros of Ai'. Odd levels
    # vanish there, at zeros of Ai. The two kinds alternate, starting with an even level.
    zeros, derivative_zeros, _, _ = ai_zeros(1)
    energies = task.hamiltonian(grid).eigenvalues[:2]
    expected = -scale * np.array([derivative_zeros[0], zeros[0]])
    assert energies == pytest.approx(expected, rel=1e-3)


def test_falling_packet_follows_the_classical_parabola_until_it_reaches_the_object():
    assert_follows_classical(Gravity(mass=1.0, g=0.5), x0=-15.0, p0=0.0, times=(0.0, 2.0, 4.0))


def test_classical_particle_falls_through_the_object_and_conserves_energy():
    task, x0, p0 = Gravity(mass=2.0, g=0.5), -6.0, 1.0
    energy = task.classical_energy(x0, p0)
    positions = []
    for time in np.linspace(0.0, 40.0, 81):
        position, momentum = task.classical_state(x0, p0, time)
        positions.append(position)
        assert task.classical_energy(position, momentum) == pytest.approx(energy)
    assert min(positions) < 0 < max(positions)
    assert task.classical_state(x0, p0, 0.0) == pytest.approx((x0, p0))


def test_equal_tasks_share_a_hamiltonian():
    first, second = HarmonicOscillator(mass=2.0), HarmonicOscillator(mass=2.0)
    assert first.hamiltonian(grid) is second.hamiltonian(grid)
