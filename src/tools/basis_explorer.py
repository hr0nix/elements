from collections.abc import Callable, Sequence
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoLocator, ScalarFormatter
from matplotlib.widgets import RadioButtons, Slider

from elements.quantum.continuous_approx import CyclicalGrid1DStateSpace, Operators1D
from elements.quantum.tasks import FreeParticle, Gravity, HarmonicOscillator, Task

LIST_ROWS = 40
SCROLL_ROWS = 3
# In list coordinates, which run from 0 to 1 across the list.
SCROLLBAR_LEFT = 0.92
KEY_STEPS = {"down": 1, "up": -1, "pagedown": LIST_ROWS, "pageup": -LIST_ROWS}
# Diagonalizing a Hamiltonian takes about a tenth of a second, too long to redo for every step
# of a slider drag, so a drag rebuilds once it pauses.
REBUILD_DELAY_MS = 150
POTENTIAL_COLOR = "tab:green"
SELECTION_COLOR = "tab:cyan"
ENVELOPE_STYLE = {"color": "gray", "linewidth": 0.8}
CURVE_STYLE = {"linestyle": "-", "marker": "None"}
# Energy levels are separate points rather than samples of a curve. Joined up, neighbouring
# levels that look nothing alike, such as an even and an odd state, fill the plot in solid.
LEVEL_STYLE = {"linestyle": "None", "marker": ".", "markersize": 3}


class Basis(NamedTuple):
    name: str
    operator: str
    # Of the eigenvalue, and of a generic eigenvector inside a bra or ket.
    symbol: str
    label: str
    # Position and momentum eigenvalues sample a continuum at even spacing; energy levels do not.
    continuous: bool


BASES = {
    basis.name: basis
    for basis in (
        Basis("Position", "X", "x", "x", continuous=True),
        Basis("Momentum", "P", "p", "p", continuous=True),
        Basis("Energy", "H", "E", "E_n", continuous=False),
    )
}


class EigenvectorList:
    # Shows a window of rows labels; row 0 is at the top. Clicking a label calls on_select with
    # its index. Scrolling redraws only the list: redrawing the whole figure takes long enough
    # to make scrolling lag.
    def __init__(self, axes: Axes, rows: int, on_select: Callable[[int], None]) -> None:
        self.axes = axes
        self.rows = rows
        self.on_select = on_select
        self.labels: list[str] = []
        self.selected = 0
        # Fractional, so that the small steps of a trackpad add up.
        self.top = 0.0
        # Where on the thumb the pointer holds it during a drag.
        self._grip: float | None = None
        axes.set_xlim(0, 1)
        axes.set_ylim(rows, 0)
        axes.set_xticks([])
        axes.set_yticks([])
        # Keeps the toolbar's zoom and pan off the rows.
        axes.set_navigate(False)
        self.highlight = axes.add_patch(
            Rectangle((0, 0), SCROLLBAR_LEFT, 1, color=SELECTION_COLOR, alpha=0.3)
        )
        self.texts = [
            axes.text(0.05, row + 0.5, "", va="center", family="monospace", fontsize=9)
            for row in range(rows)
        ]
        width = 1 - SCROLLBAR_LEFT
        self.track = axes.add_patch(Rectangle((SCROLLBAR_LEFT, 0), width, rows, color="0.93"))
        self.thumb = axes.add_patch(Rectangle((SCROLLBAR_LEFT, 0), width, rows, color="0.65"))
        canvas = axes.figure.canvas
        self._draw_pending = False
        self._draw_timer = canvas.new_timer(interval=0)
        self._draw_timer.single_shot = True
        self._draw_timer.add_callback(self._draw)
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("scroll_event", self._on_scroll)

    @property
    def first(self) -> int:
        return int(self.top)

    # Leaves drawing to the caller, which redraws the whole figure anyway.
    def set_labels(self, labels: list[str]) -> None:
        self.labels = labels
        self.top = self._clamp(self.top)
        self._update()

    # Scrolls only as far as it takes to bring the selected row into view. Leaves drawing to
    # the caller, like set_labels.
    def show(self, selected: int) -> None:
        self.selected = selected
        if selected < self.first:
            self.top = selected
        elif selected >= self.first + self.rows:
            self.top = selected - self.rows + 1
        self._update()

    def scroll(self, rows: float) -> None:
        self._scroll_to(self.top + rows)

    def _scroll_to(self, top: float) -> None:
        first = self.first
        self.top = self._clamp(top)
        if self.first == first:
            return
        self._update()
        # Drawing the rows takes longer than a trackpad takes to send its next scroll event,
        # so the events seen before the next pass of the event loop share one drawing.
        if not self._draw_pending:
            self._draw_pending = True
            self._draw_timer.start()

    def _draw(self) -> None:
        self._draw_pending = False
        for artist in (
            self.axes.patch,
            self.highlight,
            *self.texts,
            self.track,
            self.thumb,
            *self.axes.spines.values(),
        ):
            self.axes.draw_artist(artist)
        self.axes.figure.canvas.blit(self.axes.bbox)

    def _clamp(self, top: float) -> float:
        return float(np.clip(top, 0, max(len(self.labels) - self.rows, 0)))

    @property
    def _total(self) -> int:
        return max(len(self.labels), self.rows)

    def _update(self) -> None:
        first = self.first
        for row, text in enumerate(self.texts):
            index = first + row
            text.set_text(self.labels[index] if index < len(self.labels) else "")
        row = self.selected - first
        self.highlight.set_visible(0 <= row < self.rows)
        self.highlight.set_y(row)
        self.thumb.set_y(self.rows * first / self._total)
        self.thumb.set_height(self.rows**2 / self._total)

    def _on_press(self, event: MouseEvent) -> None:
        if event.inaxes is not self.axes or event.button != MouseButton.LEFT:
            return
        if event.xdata < SCROLLBAR_LEFT:
            self.on_select(self.first + int(event.ydata))
            return
        thumb_top, thumb_height = self.thumb.get_y(), self.thumb.get_height()
        on_thumb = thumb_top <= event.ydata <= thumb_top + thumb_height
        # Pressing the track beside the thumb brings the thumb's middle to the pointer.
        self._grip = event.ydata - thumb_top if on_thumb else thumb_height / 2
        self._drag_to(event.ydata)

    def _on_motion(self, event: MouseEvent) -> None:
        if self._grip is not None:
            # The pointer can leave the list mid-drag, where the event has no ydata.
            _, y = self.axes.transData.inverted().transform((event.x, event.y))
            self._drag_to(y)

    def _on_release(self, event: MouseEvent) -> None:
        self._grip = None

    def _drag_to(self, y: float) -> None:
        self._scroll_to((y - self._grip) / self.rows * self._total)

    def _on_scroll(self, event: MouseEvent) -> None:
        if event.inaxes is self.axes:
            self.scroll(-event.step * SCROLL_ROWS)


class BasisExplorer:
    def __init__(self, tasks: Sequence[Task], state: CyclicalGrid1DStateSpace) -> None:
        if not tasks:
            raise ValueError("Explorer needs at least one task")
        self.tasks = {task.name: task for task in tasks}
        self.task = tasks[0]
        self.state = state
        self.target = BASES["Momentum"]
        self.view = BASES["Position"]
        self.selected = 0
        self._listed: Basis | None = None
        self._build_figure()
        self.rebuild()

    def run(self) -> None:
        self.figure.canvas.manager.full_screen_toggle()
        plt.show()

    @property
    def mass(self) -> float:
        return self.mass_slider.val

    def select_task(self, name: str) -> None:
        self.task = self.tasks[name]
        self.rebuild()

    def select_target(self, name: str) -> None:
        self.target = BASES[name]
        self.rebuild()

    def select_view(self, name: str) -> None:
        self.view = BASES[name]
        self.rebuild()

    def rebuild(self) -> None:
        self.target_values, target_vectors = self._sorted_eigenbasis(self.target)
        self.view_values, view_vectors = self._sorted_eigenbasis(self.view)
        # Row i, column j is <view_i|target_j>: column j is target eigenvector j in the view basis.
        self.amplitudes = view_vectors.conj().T @ target_vectors
        if self.view.continuous:
            # Each overlap is the amplitude for a cell one eigenvalue spacing wide. Dividing by
            # the square root of the spacing turns it into an amplitude density.
            self.amplitudes /= np.sqrt(self.view_values[1] - self.view_values[0])
        # A new kind of eigenvector starts from the one nearest zero: x = 0, p = 0 or the ground
        # state. Otherwise the selection stays, so a level can be compared across tasks.
        if self.target != self._listed:
            self.selected = int(np.argmin(np.abs(self.target_values)))
            self._listed = self.target
        self.eigenvector_list.set_labels(
            [
                f"{index:>4}   {self.target.symbol} = {value:+9.3f}"
                for index, value in enumerate(self.target_values)
            ]
        )
        self._show_view_axis()
        self._show_overlaps()
        self.figure.suptitle(
            f"{self.task.name}: eigenvectors of {self.target.name.lower()} "
            f"${self.target.operator}$ in the {self.view.name.lower()} basis",
            fontsize=13,
        )
        self.select(self.selected)

    def select(self, index: int) -> None:
        self.selected = int(np.clip(index, 0, len(self.target_values) - 1))
        self.eigenvector_list.show(self.selected)
        amplitudes = self.amplitudes[:, self.selected]
        real, imaginary, envelope, negative_envelope = self.vector_lines
        real.set_data(self._view_coordinates, amplitudes.real)
        imaginary.set_data(self._view_coordinates, amplitudes.imag)
        envelope.set_data(self._view_coordinates, np.abs(amplitudes))
        negative_envelope.set_data(self._view_coordinates, -np.abs(amplitudes))
        peak = np.abs(amplitudes).max()
        self.vector_axes.set_ylim(-1.15 * peak, 1.15 * peak)
        if self.target.continuous:
            eigenvector = f"{self.target.symbol} = {self.target_values[self.selected]:+.2f}"
        else:
            eigenvector = f"E_{{{self.selected}}}"
        self.vector_axes.set_title(
            rf"{self.target.name} eigenvector $|{eigenvector}\rangle$ in the "
            f"{self.view.name.lower()} basis: {self._amplitude(eigenvector)}"
        )
        self.selection_line.set_ydata([self.selected, self.selected])
        self._show_potential()
        self.figure.canvas.draw_idle()

    # Overlaps with a basis that samples a continuum are shown divided by the square root of
    # the grid spacing, as amplitude densities.
    def _amplitude(self, eigenvector: str, magnitude: bool = False) -> str:
        amplitude = rf"\langle {self.view.label} | {eigenvector} \rangle"
        if magnitude:
            amplitude = f"|{amplitude}|"
        if self.view.continuous:
            amplitude += rf" / \sqrt{{d{self.view.symbol}}}"
        return f"${amplitude}$"

    def _sorted_eigenbasis(self, basis: Basis) -> tuple[np.ndarray, np.ndarray]:
        match basis.name:
            case "Position":
                operator = Operators1D.position(self.state)
            case "Momentum":
                operator = Operators1D.momentum(self.state)
            case _:
                operator = self.task.hamiltonian(self.state, self.mass)
        eigenvalues = operator.eigenvalues.real
        order = np.argsort(eigenvalues, kind="stable")
        vectors = np.hstack([vector.components for vector in operator.eigenvectors])[:, order]
        if not vectors.imag.any():
            # A real eigenvector is fixed only up to its sign. Making the lobe furthest to the
            # right positive, as for the textbook oscillator states, stops the sign jumping
            # between levels at random.
            magnitudes = np.abs(vectors.real)
            significant = magnitudes > 1e-3 * magnitudes.max(axis=0)
            rightmost = len(vectors) - 1 - np.argmax(significant[::-1], axis=0)
            vectors = vectors * np.sign(vectors.real[rightmost, np.arange(len(vectors))])
        return eigenvalues[order], vectors

    # The overlap image shares its horizontal axis with the plot above, so both follow the
    # view basis: its eigenvalues when they sample a continuum, its level numbers otherwise.
    def _show_view_axis(self) -> None:
        values = self.view_values
        axes = self.vector_axes
        if self.view.continuous:
            self._view_coordinates = values
            half_step = (values[1] - values[0]) / 2
            self._view_extent = (values[0] - half_step, values[-1] + half_step)
            axes.xaxis.set_major_locator(AutoLocator())
            axes.xaxis.set_major_formatter(ScalarFormatter())
            self.overlap_axes.set_xlabel(f"${self.view.symbol}$")
        else:
            levels = np.arange(len(values))
            self._view_coordinates = levels
            self._view_extent = (-0.5, len(values) - 0.5)
            ticks = levels[:: max(1, len(levels) // 8)]
            axes.set_xticks(ticks, labels=[f"{values[level]:.2f}" for level in ticks])
            self.overlap_axes.set_xlabel("energy $E_n$ of the level, one step per level")
        quantity = "amplitude density" if self.view.continuous else "amplitude"
        axes.set_ylabel(quantity)
        self.colorbar.set_label(quantity)
        style = CURVE_STYLE if self.view.continuous else LEVEL_STYLE
        for line in self.vector_lines:
            line.set(**style)
        # A legend copies the look of its lines when it is made.
        axes.legend(loc="upper right", fontsize=9, ncols=3)
        axes.set_xlim(*self._view_extent)

    def _show_overlaps(self) -> None:
        count = len(self.target_values)
        magnitudes = np.abs(self.amplitudes.T)
        self.overlap_image.set_data(magnitudes)
        self.overlap_image.set_extent((*self._view_extent, count - 0.5, -0.5))
        self.overlap_image.set_clim(0, magnitudes.max())
        self.overlap_axes.set_ylim(count - 0.5, -0.5)
        ticks = np.linspace(0, count - 1, 9).round().astype(int)
        self.overlap_axes.set_yticks(
            ticks, labels=[f"{self.target_values[tick]:.2f}" for tick in ticks]
        )
        self.overlap_axes.set_ylabel(
            f"eigenvectors of ${self.target.operator}$, labelled by ${self.target.symbol}$"
        )
        self.overlap_axes.set_title(
            f"{self._amplitude(self.target.label, magnitude=True)}, "
            "one row per eigenvector: click a row to show it"
        )

    # An energy eigenvector oscillates where its energy is above the potential and decays where
    # it is below. As in textbook pictures, the level is drawn at zero amplitude, so the
    # classical turning points are where the potential crosses the zero line.
    def _show_potential(self) -> None:
        values = self.task.potential(self.state.ticks, self.mass)
        visible = self.view.name == "Position" and bool(np.any(values != 0))
        self.potential_axes.set_visible(visible)
        if not visible:
            return
        self.potential_line.set_data(self.state.ticks, values)
        low, high = values.min(), values.max()
        showing_level = self.target.name == "Energy"
        self.level_line.set_visible(showing_level)
        if showing_level:
            energy = self.target_values[self.selected]
            self.level_line.set_ydata([energy, energy])
            # The amplitude axis is symmetric about zero, so this one is symmetric about the
            # level. Walls far higher than the level are cut off.
            reach = 1.05 * (energy - low)
            self.potential_axes.set_ylim(energy - reach, energy + reach)
        else:
            padding = 0.05 * (high - low) or 1.0
            self.potential_axes.set_ylim(low - padding, high + padding)

    def _on_click(self, event: MouseEvent) -> None:
        toolbar = self.figure.canvas.toolbar
        if event.inaxes is not self.overlap_axes or event.button != MouseButton.LEFT:
            return
        if toolbar is not None and toolbar.mode:
            return
        self.select(round(event.ydata))

    def _on_key(self, event: KeyEvent) -> None:
        if event.key in KEY_STEPS:
            self.select(self.selected + KEY_STEPS[event.key])

    def _rebuild_soon(self) -> None:
        self._rebuild_timer.stop()
        self._rebuild_timer.start()

    def _build_figure(self) -> None:
        self.figure = plt.figure(figsize=(14, 10))
        self._build_controls()
        self.list_axes = self.figure.add_axes((0.17, 0.06, 0.15, 0.84))
        self.list_axes.set_title("Eigenvectors by eigenvalue\nscroll, click or press up / down")
        self.eigenvector_list = EigenvectorList(self.list_axes, LIST_ROWS, self.select)
        self._build_vector_plot()
        self._build_overlap_plot()
        canvas = self.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_click)
        canvas.mpl_connect("key_press_event", self._on_key)
        self._rebuild_timer = canvas.new_timer(interval=REBUILD_DELAY_MS)
        self._rebuild_timer.single_shot = True
        self._rebuild_timer.add_callback(self.rebuild)

    def _build_controls(self) -> None:
        self.radio_buttons = {}
        for top, title, labels, initial, callback in (
            (0.88, "Task", list(self.tasks), self.task.name, self.select_task),
            (0.66, "Eigenvectors of", list(BASES), self.target.name, self.select_target),
            (0.44, "Shown in", list(BASES), self.view.name, self.select_view),
        ):
            axes = self.figure.add_axes((0.02, top - 0.14, 0.13, 0.14))
            axes.set_title(title, fontsize=10)
            buttons = RadioButtons(axes, labels, active=labels.index(initial))
            buttons.on_clicked(callback)
            self.radio_buttons[title] = buttons
        slider_axes = self.figure.add_axes((0.04, 0.24, 0.08, 0.025))
        self.mass_slider = Slider(slider_axes, "$m$", 0.2, 5.0, valinit=1.0)
        self.mass_slider.on_changed(lambda _: self._rebuild_soon())

    def _build_vector_plot(self) -> None:
        axes = self.vector_axes = self.figure.add_axes((0.39, 0.56, 0.54, 0.34))
        (real,) = axes.plot([], [], color="tab:blue", label="Re")
        (imaginary,) = axes.plot([], [], color="tab:orange", label="Im")
        # Underneath, so that Re and Im stay visible where hundreds of levels are drawn.
        (envelope,) = axes.plot([], [], label=r"$\pm|\cdot|$", zorder=1, **ENVELOPE_STYLE)
        (negative_envelope,) = axes.plot([], [], zorder=1, **ENVELOPE_STYLE)
        self.vector_lines = [real, imaginary, envelope, negative_envelope]
        axes.axhline(0, color="black", linewidth=0.5)
        self.potential_axes = axes.twinx()
        self.potential_axes.set_ylabel("$V(x)$", color=POTENTIAL_COLOR)
        (self.potential_line,) = self.potential_axes.plot(
            [], [], color=POTENTIAL_COLOR, alpha=0.7, label="$V(x)$"
        )
        self.level_line = self.potential_axes.axhline(
            0, color=POTENTIAL_COLOR, linestyle="--", label="level $E$, at zero amplitude"
        )
        self.potential_axes.legend(loc="upper left", fontsize=9)

    def _build_overlap_plot(self) -> None:
        axes = self.overlap_axes = self.figure.add_axes(
            (0.39, 0.07, 0.54, 0.4), sharex=self.vector_axes
        )
        self.overlap_image = axes.imshow(
            np.zeros((1, 1)), origin="upper", aspect="auto", cmap="magma"
        )
        colorbar_axes = self.figure.add_axes((0.94, 0.07, 0.01, 0.4))
        self.colorbar = self.figure.colorbar(self.overlap_image, cax=colorbar_axes)
        self.selection_line = axes.axhline(0, color=SELECTION_COLOR, linewidth=1, alpha=0.8)


def main() -> None:
    tasks = [FreeParticle(), HarmonicOscillator(), Gravity()]
    BasisExplorer(tasks, CyclicalGrid1DStateSpace(-30.0, 30.0, 601)).run()


if __name__ == "__main__":
    main()
