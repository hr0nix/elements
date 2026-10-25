import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from numpy.polynomial.hermite import hermval

from elements.quantum.continuous_approx import (
    CyclicalGrid1DStateSpace,
    FreeParticle,
    Gravity,
    HarmonicOscillator,
)
from tools.basis_explorer import LIST_ROWS, SCROLLBAR_LEFT, BasisExplorer

GRID = CyclicalGrid1DStateSpace(-15.0, 15.0, 201)


@pytest.fixture
def explorer():
    explorer = BasisExplorer([FreeParticle(), HarmonicOscillator(), Gravity()], GRID)
    yield explorer
    plt.close(explorer.figure)


def shown(explorer: BasisExplorer) -> tuple[np.ndarray, np.ndarray]:
    real, imaginary, _, _ = explorer.vector_lines
    return real.get_xdata(), real.get_ydata() + 1j * imaginary.get_ydata()


def test_momentum_eigenvectors_are_plane_waves_spread_evenly_over_position(explorer):
    # The list starts at p = 0, so three steps on is the third plane wave that fits the circle.
    explorer.select(explorer.selected + 3)
    x, amplitude = shown(explorer)
    p = 3 * 2 * np.pi / GRID.length
    assert explorer.target_values[explorer.selected] == pytest.approx(p)
    assert np.allclose(x, GRID.ticks)
    assert np.allclose(amplitude, np.exp(1j * p * x) / np.sqrt(GRID.length))


@pytest.mark.parametrize(
    "basis, spacing",
    [("Position", GRID.dx), ("Momentum", 2 * np.pi / GRID.length), ("Energy", 1.0)],
)
def test_eigenvectors_shown_in_their_own_basis_overlap_like_the_identity(explorer, basis, spacing):
    explorer.select_target(basis)
    explorer.select_view(basis)
    assert np.allclose(explorer.overlap_image.get_array(), np.eye(201) / np.sqrt(spacing))


def hermite_function(n: int, u: np.ndarray) -> np.ndarray:
    coefficients = np.zeros(n + 1)
    coefficients[n] = 1.0
    norm = np.sqrt(2.0**n * math.factorial(n) * np.sqrt(np.pi))
    return hermval(u, coefficients) * np.exp(-(u**2) / 2) / norm


# The Fourier transform of the n-th Hermite function is (-i)^n times itself, with momentum in
# units of sqrt(m omega) instead of position in units of 1 / sqrt(m omega).
@pytest.mark.parametrize("n", range(4))
def test_oscillator_levels_are_the_textbook_hermite_functions_in_position_and_momentum(explorer, n):
    explorer.select_task("Harmonic oscillator")
    explorer.select_target("Energy")
    explorer.select(n)
    m_omega = explorer.task.mass * explorer.task.omega
    x, amplitude = shown(explorer)
    assert np.allclose(amplitude, m_omega**0.25 * hermite_function(n, np.sqrt(m_omega) * x))

    explorer.select_view("Momentum")
    p, amplitude = shown(explorer)
    expected = (-1j) ** n * m_omega**-0.25 * hermite_function(n, p / np.sqrt(m_omega))
    assert np.allclose(amplitude, expected)


def test_gravity_levels_are_real_and_positive_past_their_right_turning_point(explorer):
    explorer.select_task("Gravity")
    explorer.select_target("Energy")
    assert not explorer.amplitudes.imag.any()
    for level in range(10):
        turning_point = explorer.target_values[level] / (explorer.task.mass * explorer.task.g)
        just_past = np.searchsorted(GRID.ticks, turning_point + 0.5)
        assert explorer.amplitudes[just_past, level].real > 0


def test_level_sits_at_zero_amplitude_in_front_of_the_potential(explorer):
    explorer.select_task("Harmonic oscillator")
    explorer.select_target("Energy")
    explorer.select(3)
    energy = 3.5 * explorer.task.omega
    assert explorer.level_line.get_ydata()[0] == pytest.approx(energy)
    assert sum(explorer.potential_axes.get_ylim()) / 2 == pytest.approx(energy)
    assert sum(explorer.vector_axes.get_ylim()) == pytest.approx(0.0)

    explorer.select_view("Momentum")
    assert not explorer.potential_axes.get_visible()


def test_energy_basis_is_drawn_as_a_dot_per_level_with_energies_on_the_ticks(explorer):
    explorer.select_task("Harmonic oscillator")
    explorer.select_view("Energy")
    assert explorer.vector_lines[0].get_linestyle() == "None"
    assert explorer.vector_lines[0].get_marker() == "."
    levels = explorer.vector_axes.get_xticks()[:2]
    labels = [float(label.get_text()) for label in explorer.vector_axes.get_xticklabels()[:2]]
    omega = explorer.task.omega
    assert labels == pytest.approx(omega * (levels + 0.5), abs=0.006)

    explorer.select_view("Position")
    assert explorer.vector_lines[0].get_linestyle() == "-"
    assert explorer.vector_lines[0].get_marker() == "None"
    half_tick = GRID.dx / 2
    assert explorer.vector_axes.get_xlim() == pytest.approx(
        (GRID.min_x - half_tick, GRID.max_x - half_tick)
    )


def test_a_new_kind_of_eigenvector_starts_nearest_zero_but_other_changes_keep_the_selection(
    explorer,
):
    assert explorer.target_values[explorer.selected] == 0.0
    explorer.select(explorer.selected + 5)
    chosen = explorer.selected
    explorer.select_view("Energy")
    explorer.select_task("Gravity")
    assert explorer.selected == chosen

    explorer.select_target("Position")
    assert abs(explorer.target_values[explorer.selected]) < GRID.dx


def mouse(explorer: BasisExplorer, name: str, axes, x: float, y: float, **kwargs) -> None:
    canvas = explorer.figure.canvas
    event = MouseEvent(name, canvas, *axes.transData.transform((x, y)), **kwargs)
    canvas.callbacks.process(name, event)


def press(explorer: BasisExplorer, axes, x: float, y: float) -> None:
    mouse(explorer, "button_press_event", axes, x, y, button=MouseButton.LEFT)


def key(explorer: BasisExplorer, name: str) -> None:
    canvas = explorer.figure.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, name))


def test_clicks_keys_and_scrolling_pick_eigenvectors_and_the_list_follows(explorer):
    listing = explorer.eigenvector_list
    explorer.select_target("Position")
    first = listing.first
    press(explorer, explorer.list_axes, 0.5, 2.5)
    assert explorer.selected == first + 2
    key(explorer, "down")
    assert explorer.selected == first + 3

    press(explorer, explorer.overlap_axes, 0.0, 180.2)
    assert explorer.selected == 180
    assert listing.first <= 180 < listing.first + listing.rows
    assert listing.highlight.get_visible()

    mouse(explorer, "scroll_event", explorer.list_axes, 0.5, 5.0, step=10)
    assert not listing.highlight.get_visible()
    key(explorer, "up")
    assert explorer.selected == 179
    assert listing.highlight.get_visible()


def test_scrollbar_jumps_to_a_press_beside_the_thumb_and_follows_a_drag(explorer):
    listing = explorer.eigenvector_list
    axes = explorer.list_axes
    bar = (1 + SCROLLBAR_LEFT) / 2
    explorer.select_target("Position")
    selected = explorer.selected

    press(explorer, axes, bar, LIST_ROWS - 0.1)
    assert listing.first == len(listing.labels) - LIST_ROWS
    assert explorer.selected == selected
    # Dragging past the end of the list stops at its first row.
    mouse(explorer, "motion_notify_event", axes, bar, -5.0)
    assert listing.first == 0
    mouse(explorer, "button_release_event", axes, bar, -5.0, button=MouseButton.LEFT)

    # Grabbed by its top, the thumb moves with the pointer: 10 rows of the bar span a quarter
    # of the 201 eigenvectors.
    press(explorer, axes, bar, 0.0)
    mouse(explorer, "motion_notify_event", axes, bar, 10.0)
    assert listing.first == int(10.0 / LIST_ROWS * 201)
    mouse(explorer, "button_release_event", axes, bar, 10.0, button=MouseButton.LEFT)
    mouse(explorer, "motion_notify_event", axes, bar, 30.0)
    assert listing.first == int(10.0 / LIST_ROWS * 201)


def test_titles_and_colorbar_say_when_overlaps_are_amplitude_densities(explorer):
    explorer.select_target("Energy")
    assert r"\sqrt{dx}" in explorer.overlap_axes.get_title()
    assert r"\sqrt{dx}" in explorer.vector_axes.get_title()
    assert explorer.colorbar.ax.get_ylabel() == "amplitude density"
    explorer.select_view("Momentum")
    assert r"\sqrt{dp}" in explorer.overlap_axes.get_title()

    explorer.select_target("Position")
    explorer.select_view("Energy")
    assert "sqrt" not in explorer.overlap_axes.get_title()
    assert explorer.colorbar.ax.get_ylabel() == "amplitude"


def test_energy_levels_follow_the_mass_once_the_slider_rebuilds(explorer):
    explorer.select_task("Harmonic oscillator")
    explorer.select_target("Energy")
    explorer.mass_slider.set_val(4.0)
    # A GUI timer runs this once the slider stops moving; the test backend has no timers.
    explorer.rebuild()
    omega = np.sqrt(explorer.task.spring_constant / 4.0)
    assert explorer.target_values[:3] == pytest.approx(omega * (np.arange(3) + 0.5))
    assert explorer.eigenvector_list.texts[0].get_text().endswith(f"{omega / 2:+9.3f}")
