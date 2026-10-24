import numpy as np
import pytest

from elements.quantum.continuous_approx import (
    CyclicalGrid1DStateSpace,
    Operators1D,
    WaveFunctions1D,
)


def test_ticks_go_round_the_circle_without_repeating_the_seam():
    grid = CyclicalGrid1DStateSpace(-1.0, 1.0, 9)
    assert grid.dx == pytest.approx(2.0 / 9)
    assert grid.ticks[0] == pytest.approx(-1.0)
    assert grid.ticks[-1] == pytest.approx(1.0 - 2.0 / 9)


@pytest.mark.parametrize("num_points", [0, 8])
def test_number_of_points_must_be_positive_and_odd(num_points):
    with pytest.raises(ValueError, match="positive and odd"):
        CyclicalGrid1DStateSpace(-1.0, 1.0, num_points)


def test_labels_stay_unique_on_fine_grids():
    assert CyclicalGrid1DStateSpace(-10.0, 10.0, 5001).dimension == 5001


def test_operators_built_from_plane_waves_have_them_as_eigenvectors():
    grid = CyclicalGrid1DStateSpace(-3.0, 3.0, 15)
    for operator in (
        Operators1D.momentum(grid),
        Operators1D.momentum_squared(grid),
        Operators1D.kinetic_energy(grid, mass=2.0),
    ):
        assert operator.is_hermitian
        for value, vector in zip(operator.eigenvalues, operator.eigenbasis, strict=True):
            assert (operator @ vector).close_to(vector * value)


def test_plane_waves_that_fit_on_the_circle_have_exact_momentum():
    grid = CyclicalGrid1DStateSpace(0.0, 5.0, 33)
    p = 2 * np.pi * 3 / grid.length
    wave = grid.sample(lambda x: np.exp(1j * p * x)).normalized
    assert (Operators1D.momentum(grid) @ wave).close_to(p * wave)
    assert (Operators1D.momentum_squared(grid) @ wave).close_to(p**2 * wave)


def test_momentum_differentiates_smooth_periodic_functions():
    grid = CyclicalGrid1DStateSpace(0.0, 2 * np.pi, 65)
    f = grid.sample(lambda x: np.exp(np.sin(x)))
    derivative = grid.sample(lambda x: np.cos(x) * np.exp(np.sin(x)))
    assert (Operators1D.momentum(grid) @ f).close_to(-1j * derivative, tolerance=1e-10)


def test_momentum_squared_is_momentum_applied_twice():
    grid = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    p = Operators1D.momentum(grid)
    assert (p @ p).close_to(Operators1D.momentum_squared(grid))


def test_translation_by_whole_ticks_shifts_any_state_cyclically():
    grid = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    rng = np.random.default_rng(0)
    values = rng.normal(size=15) + 1j * rng.normal(size=15)
    shifted = Operators1D.translation(grid, 3 * grid.dx) @ grid.basis.ket(values)
    assert shifted.close_to(grid.basis.ket(np.roll(values, 3)))


def test_translation_all_the_way_round_is_the_identity():
    grid = CyclicalGrid1DStateSpace(-2.0, 2.0, 15)
    identity = Operators1D.translation(grid, 0.0)
    assert Operators1D.translation(grid, grid.length).close_to(identity)


def test_boost_by_an_allowed_momentum_turns_plane_waves_into_plane_waves():
    grid = CyclicalGrid1DStateSpace(0.0, 5.0, 33)
    step = 2 * np.pi / grid.length
    wave = grid.sample(lambda x: np.exp(1j * 3 * step * x)).normalized
    boosted = grid.sample(lambda x: np.exp(1j * 5 * step * x)).normalized
    assert (Operators1D.boost(grid, 2 * step) @ wave).close_to(boosted)


def test_gaussian_at_rest_sits_opposite_the_seam_with_no_momentum():
    grid = CyclicalGrid1DStateSpace(0.0, 30.0, 301)
    at_rest = WaveFunctions1D.gaussian_at_rest(grid, sigma=1.5)
    assert np.allclose(at_rest.components.imag, 0.0)
    assert Operators1D.position(grid).expectation(at_rest) == pytest.approx(15.0)
    assert Operators1D.momentum(grid).expectation(at_rest) == pytest.approx(0.0)


def test_packet_is_the_closed_form_gaussian_up_to_a_global_phase():
    grid = CyclicalGrid1DStateSpace(-20.0, 20.0, 401)
    packet = WaveFunctions1D.packet(grid, x0=-3.0, p0=2.0, sigma=1.0)
    closed_form = grid.sample(lambda x: np.exp(-((x + 3.0) ** 2) / 4 + 2j * x)).normalized
    assert abs(closed_form.dagger @ packet) == pytest.approx(1.0)


def test_gaussian_packet_has_requested_centres_and_minimal_uncertainty():
    grid = CyclicalGrid1DStateSpace(-20.0, 20.0, 513)
    packet = WaveFunctions1D.packet(grid, x0=-3.0, p0=2.0, sigma=1.0)
    x, p = Operators1D.position(grid), Operators1D.momentum(grid)

    assert packet.is_normalized
    assert x.expectation(packet) == pytest.approx(-3.0)
    assert p.expectation(packet) == pytest.approx(2.0)
    spread_x = np.sqrt((x @ x).expectation(packet) - x.expectation(packet) ** 2)
    spread_p = np.sqrt((p @ p).expectation(packet) - p.expectation(packet) ** 2)
    assert spread_x == pytest.approx(1.0)
    assert spread_p == pytest.approx(0.5)


def test_packet_on_the_seam_is_a_rotated_packet_from_inside_the_grid():
    grid = CyclicalGrid1DStateSpace(-10.0, 10.0, 201)
    at_seam = WaveFunctions1D.packet(grid, x0=grid.min_x, p0=1.5, sigma=0.8)
    inside = WaveFunctions1D.packet(grid, x0=grid.ticks[100], p0=1.5, sigma=0.8)
    assert np.allclose(at_seam.components, np.roll(inside.components, -100))


def test_position_wavefunction_is_the_normalized_continuum_gaussian():
    grid = CyclicalGrid1DStateSpace(-20.0, 20.0, 401)
    packet = WaveFunctions1D.packet(grid, x0=-3.0, p0=2.0, sigma=1.0)
    x, psi = WaveFunctions1D.position_wavefunction(packet)
    assert np.allclose(x, grid.ticks)
    assert np.sum(np.abs(psi) ** 2) * grid.dx == pytest.approx(1.0)
    expected = (2 * np.pi) ** -0.25 * np.exp(-((x + 3.0) ** 2) / 4)
    assert np.allclose(np.abs(psi), expected)


def test_momentum_wavefunction_is_the_fourier_transform_of_the_packet():
    grid = CyclicalGrid1DStateSpace(-20.0, 20.0, 401)
    packet = WaveFunctions1D.packet(grid, x0=-3.0, p0=2.0, sigma=1.0)
    p, phi = WaveFunctions1D.momentum_wavefunction(packet)
    assert np.all(np.diff(p) > 0)
    assert np.sum(np.abs(phi) ** 2) * (p[1] - p[0]) == pytest.approx(1.0)
    expected = (2 / np.pi) ** 0.25 * np.exp(-((p - 2.0) ** 2))
    assert np.allclose(np.abs(phi), expected)


def test_projections_on_momentum_eigenvectors_are_an_fft_up_to_a_phase():
    grid = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    rng = np.random.default_rng(1)
    components = rng.normal(size=21) + 1j * rng.normal(size=21)
    p, phi = WaveFunctions1D.momentum_wavefunction(grid.basis.ket(components).normalized)
    fourier = np.fft.fftshift(np.fft.fft(components / np.linalg.norm(components), norm="ortho"))
    dp = 2 * np.pi / grid.length
    assert np.allclose(phi * np.sqrt(dp), np.exp(-1j * p * grid.min_x) * fourier)


def test_wavefunctions_need_a_hermitian_operator_with_evenly_spaced_eigenvalues():
    grid = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    packet = WaveFunctions1D.packet(grid, 0.5, 1.0, 0.7)
    with pytest.raises(ValueError, match="Hermitian"):
        WaveFunctions1D.wavefunction(Operators1D.translation(grid, 0.3), packet)
    with pytest.raises(ValueError, match="evenly spaced"):
        WaveFunctions1D.wavefunction(Operators1D.momentum_squared(grid), packet)


def test_factories_hand_back_the_same_object_for_equal_arguments():
    grid = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    same_grid = CyclicalGrid1DStateSpace(-3.0, 4.0, 21)
    assert Operators1D.momentum(grid) is Operators1D.momentum(same_grid)
    assert Operators1D.kinetic_energy(grid, 2.0) is Operators1D.kinetic_energy(grid, 2.0)
    assert WaveFunctions1D.packet(grid, 0.5, 1.0, 0.7) is WaveFunctions1D.packet(
        grid, 0.5, 1.0, 0.7
    )
    assert Operators1D.kinetic_energy(grid, 2.0) is not Operators1D.kinetic_energy(grid, 3.0)


def test_harmonic_oscillator_energies_are_half_integers():
    grid = CyclicalGrid1DStateSpace(-10.0, 10.0, 257)
    hamiltonian = Operators1D.hamiltonian(grid, mass=1.0, v=lambda x: 0.5 * x**2)
    assert hamiltonian.eigenvalues[:5] == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5], abs=1e-9)
