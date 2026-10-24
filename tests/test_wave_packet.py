import matplotlib.pyplot as plt
import numpy as np
import pytest

from elements.quantum.continuous_approx import CyclicalGrid1DStateSpace
from elements.quantum.tasks import FreeParticle, Gravity, HarmonicOscillator
from tools.wave_packet import WavePacketExplorer


@pytest.fixture
def explorer():
    explorer = WavePacketExplorer([FreeParticle()], CyclicalGrid1DStateSpace(-15.0, 15.0, 201))
    yield explorer
    plt.close(explorer.figure)


def test_expectations_track_the_classical_particle_across_the_seam(explorer):
    explorer.sliders["x0"].set_val(8.0)
    explorer.sliders["p0"].set_val(4.0)
    explorer.advance()
    for _ in range(100):
        explorer.advance(seconds=0.05)
    explorer.figure.canvas.draw()

    history = explorer.history
    assert explorer.time == pytest.approx(5.0)
    unwrapped = np.unwrap(np.array(history["x"]), period=30.0)
    assert unwrapped == pytest.approx(history["classical_x"], abs=1e-6)
    assert history["p"] == pytest.approx(history["classical_p"])


def test_pausing_freezes_time_and_changing_a_slider_starts_again(explorer):
    explorer.advance()
    explorer.advance()
    explorer.toggle_play()
    paused_at = explorer.time
    explorer.advance()
    assert explorer.time == paused_at
    assert explorer.play_button.label.get_text() == "Play"

    explorer.sliders["sigma"].set_val(2.0)
    explorer.advance()
    assert explorer.time == 0.0
    assert explorer.history["time"] == [0.0]


@pytest.mark.parametrize("task", [HarmonicOscillator(), Gravity()], ids=lambda task: task.name)
def test_bound_tasks_start_out_following_their_classical_particle(task):
    explorer = WavePacketExplorer(
        [FreeParticle(), task], CyclicalGrid1DStateSpace(-15.0, 15.0, 201)
    )
    explorer.select_task(task.name)
    # Far enough from x = 0 that no part of the packet has crossed the object yet.
    explorer.sliders["x0"].set_val(-10.0)
    explorer.advance()
    for _ in range(20):
        explorer.advance(seconds=0.05)
    explorer.figure.canvas.draw()
    plt.close(explorer.figure)

    history = explorer.history
    assert history["x"][-1] == pytest.approx(history["classical_x"][-1], abs=1e-4)
    assert history["p"][-1] == pytest.approx(history["classical_p"][-1], abs=1e-4)


def test_levels_stay_in_step_for_the_oscillator_but_drift_apart_under_gravity():
    grid = CyclicalGrid1DStateSpace(-30.0, 30.0, 301)
    explorer = WavePacketExplorer([HarmonicOscillator(), Gravity()], grid)
    explorer.advance()
    assert explorer.coherence == pytest.approx(1.0)
    for _ in range(30):
        explorer.advance(seconds=10.0)
    assert explorer.coherence == pytest.approx(1.0, abs=1e-6)

    explorer.select_task("Gravity")
    explorer.advance()
    for _ in range(30):
        explorer.advance(seconds=10.0)
    explorer.figure.canvas.draw()
    plt.close(explorer.figure)
    assert explorer.coherence < 0.3


def test_energy_bars_are_labelled_with_level_energies_that_follow_the_mass():
    oscillator = HarmonicOscillator()
    explorer = WavePacketExplorer([oscillator], CyclicalGrid1DStateSpace(-30.0, 30.0, 301))
    for mass in (1.0, 4.0):
        explorer.sliders["mass"].set_val(mass)
        explorer.advance()
        levels = explorer.energy_axes.get_xticks()
        labels = [float(label.get_text()) for label in explorer.energy_axes.get_xticklabels()]
        omega = np.sqrt(oscillator.spring_constant / mass)
        assert labels == pytest.approx(omega * (levels + 0.5), abs=0.006)
    plt.close(explorer.figure)


def test_energy_markers_sit_at_the_classical_energy_and_half_a_level_above_it():
    oscillator = HarmonicOscillator()
    explorer = WavePacketExplorer([oscillator], CyclicalGrid1DStateSpace(-30.0, 30.0, 301))
    explorer.sliders["x0"].set_val(-8.0)
    explorer.advance()
    plt.close(explorer.figure)

    omega = np.sqrt(oscillator.spring_constant / 1.0)
    classical_energy = 2.0**2 / 2 + 0.5 * oscillator.spring_constant * 8.0**2
    classical_level = explorer.classical_energy_line.get_xdata()[0]
    mean_level = explorer.mean_energy_line.get_xdata()[0]
    assert classical_level == pytest.approx(classical_energy / omega - 0.5, abs=1e-6)
    # A coherent state's spread adds exactly the zero-point energy omega / 2.
    assert mean_level - classical_level == pytest.approx(0.5, abs=1e-6)


def test_escape_momentum_is_marked_for_gravity_but_not_for_a_free_particle(explorer):
    explorer.advance()
    assert explorer.bound_text.get_text() == ""
    assert not any(mark.get_visible() for mark in explorer.bound_marks)

    gravity = Gravity(g=0.5)
    explorer = WavePacketExplorer([gravity], CyclicalGrid1DStateSpace(-15.0, 15.0, 201))
    explorer.sliders["x0"].set_val(-5.0)
    explorer.advance()
    plt.close(explorer.figure)
    escape = np.sqrt(2 * 1.0 * gravity.g * (15.0 - 5.0))
    assert f"{escape:.2f}" in explorer.bound_text.get_text()
    assert [mark.get_xdata()[0] for mark in explorer.bound_marks] == pytest.approx(
        [-escape, escape]
    )
    assert all(mark.get_visible() for mark in explorer.bound_marks)
