import time
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.widgets import Button, Slider

from elements.quantum.continuous_approx import (
    CyclicalGrid1DStateSpace,
    Operators1D,
    WaveFunctions1D,
)
from elements.quantum.evolution import EigenbasisEvolution
from elements.quantum.state import StateSpaceVector
from elements.quantum.tasks import FreeParticle, Gravity, HarmonicOscillator, Task
from tools.plotting import POTENTIAL_COLOR, radio_buttons, show_wavefunction, wavefunction_lines

FRAME_SECONDS = 0.03
# A stall (window dragged, laptop asleep) should not make the packet jump ahead.
LONGEST_STEP_SECONDS = 0.1
MOMENTUM_RANGE = 10.0
HISTORY_SPAN = 10.0
DENSITY_COLOR = "tab:blue"
QUANTUM_STYLE = {"color": "tab:red", "linewidth": 1.5}
CLASSICAL_STYLE = {"color": "black", "linestyle": "--", "linewidth": 1.2}
# Cyclic, dark at phase 0 and light half a turn away.
PHASE_COLORMAP = plt.get_cmap("twilight")


class WavePacketExplorer:
    def __init__(self, tasks: Sequence[Task], grid: CyclicalGrid1DStateSpace) -> None:
        if not tasks:
            raise ValueError("Explorer needs at least one task")
        self.tasks = {task.name: task for task in tasks}
        self.task = tasks[0]
        self.grid = grid
        self.momentum = Operators1D.momentum(grid)
        # exp(2 pi i X / L), the boost by one quantum of momentum: the one function of X that
        # turns smoothly all the way round the ring.
        self._ring_boost = Operators1D.boost(grid, 2 * np.pi / grid.length)
        self.playing = True
        self.time = 0.0
        self.history: dict[str, list[float]] = {}
        self.coherence = 1.0
        self.animation: FuncAnimation | None = None
        self._restart_pending = True
        self._last_frame: float | None = None
        self._build_figure()

    def run(self) -> None:
        # Kept on self: an animation stops once its object is garbage collected.
        self.animation = FuncAnimation(
            self.figure,
            lambda _: self.advance(self._seconds_since_last_frame()),
            interval=int(FRAME_SECONDS * 1000),
            cache_frame_data=False,
        )
        # Full screen rather than maximized: macOS has no separate maximize, and every backend
        # can go full screen. Matplotlib's f key toggles it back.
        self.figure.canvas.manager.full_screen_toggle()
        plt.show()

    # Moves the simulation on by speed * seconds of simulated time and redraws.
    def advance(self, seconds: float = FRAME_SECONDS) -> None:
        if self._restart_pending:
            self._restart()
        elif self.playing:
            self.time += self.sliders["speed"].val * seconds
        else:
            return
        self._show(self.evolution.state_at(self.time))

    def _seconds_since_last_frame(self) -> float:
        now = time.perf_counter()
        seconds = FRAME_SECONDS if self._last_frame is None else now - self._last_frame
        self._last_frame = now
        return min(seconds, LONGEST_STEP_SECONDS)

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")

    # Slider callbacks can fire many times per drag, so the expensive rebuild waits for the
    # next frame.
    def restart(self) -> None:
        self._restart_pending = True

    def select_task(self, name: str) -> None:
        self.task = self.tasks[name]
        self.restart()

    def _restart(self) -> None:
        values = {name: slider.val for name, slider in self.sliders.items()}
        self._initial = (values["x0"], values["p0"], values["mass"])
        packet = WaveFunctions1D.packet(self.grid, values["x0"], values["p0"], values["sigma"])
        hamiltonian = self.task.hamiltonian(self.grid, values["mass"])
        self.evolution = EigenbasisEvolution(hamiltonian, packet)
        self.time = 0.0
        self.history = {key: [] for key in ("time", "x", "p", "classical_x", "classical_p")}
        x0, p0, mass = self._initial
        energy = hamiltonian.expectation(packet)
        classical_energy = p0**2 / (2 * mass) + self.task.potential(np.array([x0]), mass)[0]
        self._build_energy_bars(energy, classical_energy)
        self._rescale(packet)
        self._show_potential(mass, energy)
        self._show_escape_momentum(x0, mass)
        self._restart_pending = False

    def _show(self, state: StateSpaceVector) -> None:
        x, psi = WaveFunctions1D.position_wavefunction(state)
        p, phi = WaveFunctions1D.momentum_wavefunction(state)
        density = np.abs(psi) ** 2
        mean_x = self._ring_mean(state)
        mean_p = self.momentum.expectation(state)
        x0, p0, mass = self._initial
        classical_x, classical_p = self.task.classical(x0, p0, mass, self.time)
        wrapped_x = self._wrap(classical_x)
        for key, value in (
            ("time", self.time),
            ("x", mean_x),
            ("p", mean_p),
            ("classical_x", classical_x),
            ("classical_p", classical_p),
        ):
            self.history[key].append(value)

        self.density_line.set_data(x, density)
        self.density_fill.set_data(x, 0, density)
        show_wavefunction(self.position_lines, x, psi)
        show_wavefunction(self.momentum_lines, p, phi)
        for line in self.mean_x_lines:
            line.set_xdata([mean_x, mean_x])
        for line in self.classical_x_lines:
            line.set_xdata([wrapped_x, wrapped_x])
        self.mean_p_line.set_xdata([mean_p, mean_p])
        self.classical_p_line.set_xdata([classical_p, classical_p])
        self._grow(self.density_axes, density.max(), symmetric=False)
        self._grow(self.position_axes, np.abs(psi).max(), symmetric=True)
        self._grow(self.momentum_axes, np.abs(phi).max(), symmetric=True)
        self._show_history()
        self._show_energy_drift()
        self.figure.suptitle(
            f"{self.task.name}      $t = {self.time:.2f}$      "
            rf"$\langle x \rangle = {mean_x:+.2f}$ (classical ${wrapped_x:+.2f}$)      "
            rf"$\langle p \rangle = {mean_p:+.2f}$ (classical ${classical_p:+.2f}$)",
            fontsize=13,
        )

    # Bars for the levels that hold the packet, ordered by energy. Their heights |c_n|^2 never
    # change, so they are built once per restart. The best evenly spaced ladder a + b n through
    # the energies is what would make the motion exactly periodic; only the departures from it
    # are kept for colouring the phases.
    def _build_energy_bars(self, energy: float, classical_energy: float) -> None:
        energies = self.evolution.energies
        weights = np.abs(self.evolution.energy_components) ** 2
        held = np.flatnonzero(weights > 1e-3 * weights.max())
        levels = np.arange(max(held[0] - 2, 0), min(held[-1] + 3, len(weights)))
        self._level_weights = weights[levels]
        if len(held) > 1:
            slope, intercept = np.polyfit(
                levels, energies[levels], 1, w=np.sqrt(self._level_weights)
            )
            self._ladder_departures = energies[levels] - (intercept + slope * levels)
        else:
            self._ladder_departures = np.zeros(len(levels))
        if self.energy_bars is not None:
            self.energy_bars.remove()
        self.energy_bars = self.energy_axes.bar(levels, self._level_weights, width=0.85)
        self.energy_axes.set_xlim(levels[0] - 1, levels[-1] + 1)
        self.energy_axes.set_ylim(0, 1.15 * self._level_weights.max())
        # Bars stay one per level, which keeps degenerate levels apart; the ticks give the level
        # energies, which move with the mass.
        ticks = levels[:: max(1, len(levels) // 8)]
        self.energy_axes.set_xticks(ticks, labels=[f"{level:.2f}" for level in energies[ticks]])
        # The axis counts levels, so an energy is placed between the levels around it.
        for line, value in (
            (self.mean_energy_line, energy),
            (self.classical_energy_line, classical_energy),
        ):
            position = np.interp(value, energies[levels], levels)
            line.set_xdata([position, position])

    # Coherence is 1 while every level keeps in step with the ladder, as in the oscillator, and
    # falls as their phases drift apart: the packet spreads or its oscillation collapses.
    def _show_energy_drift(self) -> None:
        drift = np.angle(np.exp(-1j * self._ladder_departures * self.time))
        for bar, phase in zip(self.energy_bars, drift, strict=True):
            bar.set_color(PHASE_COLORMAP((phase + np.pi) / (2 * np.pi)))
        weights = self._level_weights
        self.coherence = float(np.abs(np.sum(weights * np.exp(1j * drift))) / weights.sum())
        self.coherence_text.set_text(f"coherence {self.coherence:.2f}")

    def _show_history(self) -> None:
        times = np.array(self.history["time"])
        # The ring position jumps by a circumference at the seam. Unwrapping follows the packet
        # onto the line, where the classical particle lives.
        quantum_x = np.unwrap(np.array(self.history["x"]), period=self.grid.length)
        self.x_history_lines[0].set_data(times, quantum_x)
        self.x_history_lines[1].set_data(times, self.history["classical_x"])
        self.p_history_lines[0].set_data(times, self.history["p"])
        self.p_history_lines[1].set_data(times, self.history["classical_p"])
        self.x_history_axes.set_xlim(0, max(HISTORY_SPAN, self.time))
        self._fit(self.x_history_axes, quantum_x, self.history["classical_x"])
        self._fit(self.p_history_axes, self.history["p"], self.history["classical_p"])

    # X jumps at the seam, so <X> stops meaning anything once the packet reaches it. The
    # circular mean, the angle of <exp(2 pi i X / L)> scaled to the circumference, equals <X>
    # for a symmetric packet away from the seam and keeps tracking the packet across it.
    def _ring_mean(self, state: StateSpaceVector) -> float:
        turn = np.angle(state.dagger @ self._ring_boost @ state) / (2 * np.pi)
        return self._wrap(self.grid.length * turn)

    def _wrap(self, x: float) -> float:
        return self.grid.min_x + (x - self.grid.min_x) % self.grid.length

    @staticmethod
    def _fit(axes: Axes, *series: Sequence[float], minimum_span: float = 1.0) -> None:
        values = np.concatenate([np.asarray(values, dtype=float) for values in series])
        middle = (values.min() + values.max()) / 2
        half = max(0.55 * (values.max() - values.min()), minimum_span / 2)
        axes.set_ylim(middle - half, middle + half)

    def _rescale(self, packet: StateSpaceVector) -> None:
        peak_psi = np.abs(WaveFunctions1D.position_wavefunction(packet)[1]).max()
        peak_phi = np.abs(WaveFunctions1D.momentum_wavefunction(packet)[1]).max()
        self.density_axes.set_ylim(0, 1.15 * peak_psi**2)
        self.position_axes.set_ylim(-1.15 * peak_psi, 1.15 * peak_psi)
        self.momentum_axes.set_ylim(-1.15 * peak_phi, 1.15 * peak_phi)

    # A packet that breathes or refocuses can outgrow the scale it started with. Axes only ever
    # grow, so the picture does not jitter from frame to frame.
    @staticmethod
    def _grow(axes: Axes, peak: float, symmetric: bool) -> None:
        top = axes.get_ylim()[1]
        if peak > top / 1.05:
            top = 1.15 * peak
            axes.set_ylim(-top if symmetric else 0, top)

    def _show_potential(self, mass: float, energy: float) -> None:
        values = self.task.potential(self.grid.ticks, mass)
        visible = bool(np.any(values != 0))
        self.potential_axes.set_visible(visible)
        if visible:
            self.potential_line.set_ydata(values)
            self.energy_line.set_ydata([energy, energy])
            # Walls can be far higher than anything the packet reaches. Showing the potential up
            # to twice the packet's energy keeps the classical turning points in view.
            low = values.min()
            high = min(values.max(), low + 2 * (energy - low))
            padding = 0.05 * (high - low) or 1.0
            self.potential_axes.set_ylim(low - padding, high + padding)

    # A classical particle stays bound when it cannot climb over the highest point of the
    # ring. Past that the line-based classical solution no longer describes it.
    def _show_escape_momentum(self, x0: float, mass: float) -> None:
        values = self.task.potential(self.grid.ticks, mass)
        climb = values.max() - self.task.potential(np.array([x0]), mass)[0]
        escape = float(np.sqrt(2 * mass * climb)) if climb > 0 else 0.0
        slider = self.sliders["p0"]
        for mark, value in zip(self.bound_marks, (-escape, escape), strict=True):
            mark.set_xdata([value, value])
            mark.set_visible(escape > 0 and slider.valmin <= value <= slider.valmax)
        self.bound_text.set_text(
            rf"classical particle stays bound for $|p_0| < {escape:.2f}$" if escape > 0 else ""
        )

    def _build_figure(self) -> None:
        self.figure = plt.figure(figsize=(14, 10))
        columns = self.figure.add_gridspec(
            1, 2, left=0.06, right=0.97, top=0.92, bottom=0.28, wspace=0.2, width_ratios=[1.5, 1]
        )
        left = columns[0].subgridspec(3, 1, hspace=0.55)
        right = columns[1].subgridspec(3, 1, hspace=0.7)
        self.density_axes = self.figure.add_subplot(left[0])
        self.position_axes = self.figure.add_subplot(left[1], sharex=self.density_axes)
        self.energy_axes = self.figure.add_subplot(left[2])
        self.momentum_axes = self.figure.add_subplot(right[0])
        self.x_history_axes = self.figure.add_subplot(right[1])
        self.p_history_axes = self.figure.add_subplot(right[2], sharex=self.x_history_axes)
        self._build_position_plots()
        self._build_energy_plot()
        self._build_momentum_plot()
        self._build_history_plots()
        self._build_controls()

    def _build_energy_plot(self) -> None:
        axes = self.energy_axes
        axes.set_title(
            r"Energy basis $|c_n|^2$, shaded by phase: dark in step with an evenly spaced ladder"
        )
        axes.set_xlabel("energy $E_n$ of the level, one bar per level")
        self.energy_bars = None
        self.mean_energy_line = axes.axvline(0, label=r"$\langle H \rangle$", **QUANTUM_STYLE)
        self.classical_energy_line = axes.axvline(0, label="classical $E$", **CLASSICAL_STYLE)
        axes.legend(loc="upper left", fontsize=8)
        self.coherence_text = axes.text(0.99, 0.85, "", transform=axes.transAxes, ha="right")

    def _build_position_plots(self) -> None:
        ticks = self.grid.ticks
        axes = self.density_axes
        axes.set_title(r"Probability density $|\psi(x)|^2$")
        (self.density_line,) = axes.plot(ticks, np.zeros_like(ticks), color=DENSITY_COLOR)
        self.density_fill = axes.fill_between(
            ticks, np.zeros_like(ticks), color=DENSITY_COLOR, alpha=0.25
        )
        axes.set_xlim(self.grid.min_x, self.grid.max_x)
        self.potential_axes = axes.twinx()
        self.potential_axes.set_ylabel("$V(x)$", color=POTENTIAL_COLOR)
        (self.potential_line,) = self.potential_axes.plot(
            ticks, np.zeros_like(ticks), color=POTENTIAL_COLOR, alpha=0.7, label="$V(x)$"
        )
        self.energy_line = self.potential_axes.axhline(
            0, color=POTENTIAL_COLOR, linestyle=":", label=r"$\langle H \rangle$"
        )
        self.potential_axes.legend(loc="upper left", fontsize=9)

        axes = self.position_axes
        axes.set_title(r"Wavefunction $\psi(x)$")
        axes.set_xlabel("$x$")
        self.position_lines = wavefunction_lines(axes, r"\psi")

        self.mean_x_lines = []
        self.classical_x_lines = []
        for axes in (self.density_axes, self.position_axes):
            mean = axes.axvline(0, label=r"$\langle x \rangle$", **QUANTUM_STYLE)
            classical = axes.axvline(0, label="classical $x$", **CLASSICAL_STYLE)
            self.mean_x_lines.append(mean)
            self.classical_x_lines.append(classical)
        self.density_axes.legend(loc="upper right", fontsize=9)
        self.position_axes.legend(loc="upper right", fontsize=9, ncols=3)

    def _build_momentum_plot(self) -> None:
        axes = self.momentum_axes
        axes.set_title(r"Momentum wavefunction $\tilde\psi(p)$")
        axes.set_xlabel("$p$")
        axes.set_xlim(-MOMENTUM_RANGE, MOMENTUM_RANGE)
        self.momentum_lines = wavefunction_lines(axes, r"\tilde\psi")
        self.mean_p_line = axes.axvline(0, label=r"$\langle p \rangle$", **QUANTUM_STYLE)
        self.classical_p_line = axes.axvline(0, label="classical $p$", **CLASSICAL_STYLE)
        # Re and Im share their colours with the position panel, so only the markers need a key.
        axes.legend(handles=[self.mean_p_line, self.classical_p_line], loc="upper left", fontsize=8)

    def _build_history_plots(self) -> None:
        self.x_history_lines = []
        self.p_history_lines = []
        for axes, symbol, lines in (
            (self.x_history_axes, "x", self.x_history_lines),
            (self.p_history_axes, "p", self.p_history_lines),
        ):
            axes.set_title(f"${symbol}$: quantum expectation vs classical")
            (quantum,) = axes.plot([], [], label=rf"$\langle {symbol} \rangle$", **QUANTUM_STYLE)
            (classical,) = axes.plot([], [], label="classical", **CLASSICAL_STYLE)
            lines.extend([quantum, classical])
            axes.legend(loc="upper left", fontsize=8)
        self.p_history_axes.set_xlabel("$t$")

    def _build_controls(self) -> None:
        reach = 0.35 * self.grid.length
        specs = {
            "x0": (
                r"$x_0$",
                self.grid.middle - reach,
                self.grid.middle + reach,
                self.grid.middle - reach / 2,
            ),
            "p0": (r"$p_0$", -5.0, 5.0, 2.0),
            "sigma": (r"$\sigma$", 0.3, 3.0, 1.0),
            "mass": ("$m$", 0.2, 5.0, 1.0),
            "speed": ("speed", 0.1, 5.0, 1.0),
        }
        self.sliders = {}
        for row, (name, (label, low, high, initial)) in enumerate(specs.items()):
            axes = self.figure.add_axes((0.08, 0.2 - 0.04 * row, 0.45, 0.025))
            slider = Slider(axes, label, low, high, valinit=initial)
            if name != "speed":
                slider.on_changed(lambda _: self.restart())
            self.sliders[name] = slider
        self.bound_marks = [
            self.sliders["p0"].ax.axvline(0, color=POTENTIAL_COLOR, linewidth=2) for _ in range(2)
        ]
        self.bound_text = self.figure.text(0.08, 0.008, "", color=POTENTIAL_COLOR, fontsize=9)

        self.task_buttons = radio_buttons(
            self.figure,
            (0.62, 0.04, 0.16, 0.16),
            "Task",
            list(self.tasks),
            self.task.name,
            self.select_task,
        )
        self.play_button = Button(self.figure.add_axes((0.83, 0.15, 0.1, 0.06)), "Pause")
        self.play_button.on_clicked(lambda _: self.toggle_play())
        self.reset_button = Button(self.figure.add_axes((0.83, 0.06, 0.1, 0.06)), "Reset")
        self.reset_button.on_clicked(lambda _: self.restart())


def main() -> None:
    tasks = [FreeParticle(), HarmonicOscillator(), Gravity()]
    WavePacketExplorer(tasks, CyclicalGrid1DStateSpace(-30.0, 30.0, 601)).run()


if __name__ == "__main__":
    main()
