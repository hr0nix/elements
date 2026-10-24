import numpy as np
import pytest

from elements.quantum.continuous_approx import (
    CyclicalGrid1DStateSpace,
    Operators1D,
    WaveFunctions1D,
)
from elements.quantum.state import StateSpaceVector


def ket(state: CyclicalGrid1DStateSpace, values: np.ndarray) -> StateSpaceVector:
    return StateSpaceVector(state, values[:, np.newaxis])


def test_ticks_go_round_the_circle_without_repeating_the_seam():
    state = CyclicalGrid1DStateSpace(-1.0, 1.0, 9)
    assert state.dx == pytest.approx(2.0 / 9)
    assert state.ticks[0] == pytest.approx(-1.0)
    assert state.ticks[-1] == pytest.approx(1.0 - 2.0 / 9)


@pytest.mark.parametrize("num_points", [0, 8])
def test_number_of_points_must_be_positive_and_odd(num_points):
    with pytest.raises(ValueError, match="positive and odd"):
        CyclicalGrid1DStateSpace(-1.0, 1.0, num_points)


def test_labels_stay_unique_on_fine_grids():
    assert CyclicalGrid1DStateSpace(-10.0, 10.0, 5001).dimension == 5001


def test_cached_eigenpairs_describe_the_matrix():
    state = CyclicalGrid1DStateSpace(-3.0, 3.0, 15)
    for operator in (
        Operators1D.momentum(state),
        Operators1D.momentum_squared(state),
        Operators1D.kinetic_energy(state, mass=2.0),
    ):
        vectors = np.hstack([vector.components for vector in operator.eigenvectors])
        assert np.allclose(operator.matrix @ vectors, vectors * operator.eigenvalues)
        assert np.allclose(vectors.conj().T @ vectors, np.eye(15))
        assert operator.is_hermitian


def test_plane_waves_that_fit_on_the_circle_have_exact_momentum():
    state = CyclicalGrid1DStateSpace(0.0, 5.0, 33)
    p = 2 * np.pi * 3 / state.length
    wave = ket(state, np.exp(1j * p * state.ticks)).normalized
    assert (Operators1D.momentum(state) @ wave).close_to(p * wave)
    assert (Operators1D.momentum_squared(state) @ wave).close_to(p**2 * wave)


def test_momentum_differentiates_smooth_periodic_functions():
    state = CyclicalGrid1DStateSpace(0.0, 2 * np.pi, 65)
    x = state.ticks
    f = ket(state, np.exp(np.sin(x)))
    derivative = ket(state, np.cos(x) * np.exp(np.sin(x)))
    assert (Operators1D.momentum(state) @ f).close_to(-1j * derivative, tolerance=1e-10)


def test_momentum_squared_is_momentum_applied_twice():
    state = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    p = Operators1D.momentum(state)
    assert (p @ p).close_to(Operators1D.momentum_squared(state))


def test_translation_is_generated_by_momentum():
    state = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    momentum = Operators1D.momentum(state)
    assert Operators1D.translation(state, 0.37).close_to(momentum.exp(-0.37j))


def test_translation_by_whole_ticks_shifts_any_state_cyclically():
    state = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    rng = np.random.default_rng(0)
    values = rng.normal(size=15) + 1j * rng.normal(size=15)
    shifted = Operators1D.translation(state, 3 * state.dx) @ ket(state, values)
    assert shifted.close_to(ket(state, np.roll(values, 3)))


def test_translation_all_the_way_round_is_the_identity():
    state = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    identity = Operators1D.translation(state, 0.0)
    assert Operators1D.translation(state, state.length).close_to(identity)


def test_boost_by_an_allowed_momentum_turns_plane_waves_into_plane_waves():
    state = CyclicalGrid1DStateSpace(0.0, 5.0, 33)
    step = 2 * np.pi / state.length
    wave = ket(state, np.exp(1j * 3 * step * state.ticks)).normalized
    boosted = ket(state, np.exp(1j * 5 * step * state.ticks)).normalized
    assert (Operators1D.boost(state, 2 * step) @ wave).close_to(boosted)


def test_gaussian_at_rest_sits_opposite_the_seam_with_no_momentum():
    state = CyclicalGrid1DStateSpace(0.0, 30.0, 301)
    at_rest = WaveFunctions1D.gaussian_at_rest(state, sigma=1.5)
    assert np.allclose(at_rest.components.imag, 0.0)
    assert Operators1D.position(state).expectation(at_rest) == pytest.approx(15.0)
    assert Operators1D.momentum(state).expectation(at_rest) == pytest.approx(0.0)


def test_packet_is_the_closed_form_gaussian_up_to_a_global_phase():
    state = CyclicalGrid1DStateSpace(-20.0, 20.0, 401)
    x = state.ticks
    packet = WaveFunctions1D.packet(state, x0=-3.0, p0=2.0, sigma=1.0)
    closed_form = ket(state, np.exp(-((x + 3.0) ** 2) / 4 + 2j * x)).normalized
    assert abs(closed_form.dagger @ packet) == pytest.approx(1.0)


def test_gaussian_packet_has_requested_centres_and_minimal_uncertainty():
    state = CyclicalGrid1DStateSpace(-20.0, 20.0, 513)
    packet = WaveFunctions1D.packet(state, x0=-3.0, p0=2.0, sigma=1.0)
    x, p = Operators1D.position(state), Operators1D.momentum(state)

    assert packet.is_normalized
    assert x.expectation(packet) == pytest.approx(-3.0)
    assert p.expectation(packet) == pytest.approx(2.0)
    spread_x = np.sqrt((x @ x).expectation(packet) - x.expectation(packet) ** 2)
    spread_p = np.sqrt((p @ p).expectation(packet) - p.expectation(packet) ** 2)
    assert spread_x == pytest.approx(1.0)
    assert spread_p == pytest.approx(0.5)


def test_packet_on_the_seam_is_a_rotated_packet_from_inside_the_grid():
    state = CyclicalGrid1DStateSpace(-10.0, 10.0, 201)
    at_seam = WaveFunctions1D.packet(state, x0=state.min_x, p0=1.5, sigma=0.8)
    inside = WaveFunctions1D.packet(state, x0=state.ticks[100], p0=1.5, sigma=0.8)
    assert np.allclose(at_seam.components, np.roll(inside.components, -100))


def test_position_wavefunction_is_the_normalized_continuum_gaussian():
    state = CyclicalGrid1DStateSpace(-20.0, 20.0, 401)
    packet = WaveFunctions1D.packet(state, x0=-3.0, p0=2.0, sigma=1.0)
    x, psi = WaveFunctions1D.position_wavefunction(packet)
    assert np.allclose(x, state.ticks)
    assert np.sum(np.abs(psi) ** 2) * state.dx == pytest.approx(1.0)
    expected = (2 * np.pi) ** -0.25 * np.exp(-((x + 3.0) ** 2) / 4)
    assert np.allclose(np.abs(psi), expected)


def test_momentum_wavefunction_is_the_fourier_transform_of_the_packet():
    state = CyclicalGrid1DStateSpace(-20.0, 20.0, 401)
    packet = WaveFunctions1D.packet(state, x0=-3.0, p0=2.0, sigma=1.0)
    p, phi = WaveFunctions1D.momentum_wavefunction(packet)
    assert np.all(np.diff(p) > 0)
    assert np.sum(np.abs(phi) ** 2) * (p[1] - p[0]) == pytest.approx(1.0)
    expected = (2 / np.pi) ** 0.25 * np.exp(-((p - 2.0) ** 2))
    assert np.allclose(np.abs(phi), expected)


def test_projections_on_momentum_eigenvectors_are_an_fft_up_to_a_phase():
    state = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    rng = np.random.default_rng(1)
    components = rng.normal(size=21) + 1j * rng.normal(size=21)
    p, phi = WaveFunctions1D.momentum_wavefunction(ket(state, components).normalized)
    fourier = np.fft.fftshift(np.fft.fft(components / np.linalg.norm(components), norm="ortho"))
    dp = 2 * np.pi / state.length
    assert np.allclose(phi * np.sqrt(dp), np.exp(-1j * p * state.min_x) * fourier)


def test_wavefunctions_need_a_hermitian_operator_with_evenly_spaced_eigenvalues():
    state = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    packet = WaveFunctions1D.packet(state, 0.5, 1.0, 0.7)
    with pytest.raises(ValueError, match="Hermitian"):
        WaveFunctions1D._wavefunction_in_eigenbasis(Operators1D.translation(state, 0.3), packet)
    with pytest.raises(ValueError, match="evenly spaced"):
        WaveFunctions1D._wavefunction_in_eigenbasis(Operators1D.momentum_squared(state), packet)


def test_factories_hand_back_the_same_object_for_equal_arguments():
    state = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    same_grid = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    assert Operators1D.momentum(state) is Operators1D.momentum(same_grid)
    assert Operators1D.kinetic_energy(state, 2.0) is Operators1D.kinetic_energy(state, 2.0)
    assert WaveFunctions1D.packet(state, 0.5, 1.0, 0.7) is WaveFunctions1D.packet(
        state, 0.5, 1.0, 0.7
    )
    assert Operators1D.kinetic_energy(state, 2.0) is not Operators1D.kinetic_energy(state, 3.0)


def test_harmonic_oscillator_energies_are_half_integers():
    state = CyclicalGrid1DStateSpace(-10.0, 10.0, 257)
    hamiltonian = Operators1D.hamiltonian(state, mass=1.0, v=lambda x: 0.5 * x**2)
    energies = np.sort(hamiltonian.eigenvalues.real)[:5]
    assert energies == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5], abs=1e-9)
