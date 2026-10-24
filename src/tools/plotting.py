from collections.abc import Callable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import RadioButtons

POTENTIAL_COLOR = "tab:green"
ENVELOPE_STYLE = {"color": "gray", "linewidth": 0.8}


# Lines for Re, Im and the envelope +-|.| of a complex function, which show_wavefunction fills
# in. The envelope goes underneath, so that Re and Im stay visible where they crowd together.
def wavefunction_lines(axes: Axes, symbol: str | None = None) -> list[Line2D]:
    name = "" if symbol is None else f" ${symbol}$"
    magnitude = r"\cdot" if symbol is None else symbol
    (real,) = axes.plot([], [], color="tab:blue", label=f"Re{name}")
    (imaginary,) = axes.plot([], [], color="tab:orange", label=f"Im{name}")
    (envelope,) = axes.plot([], [], label=rf"$\pm|{magnitude}|$", zorder=1, **ENVELOPE_STYLE)
    (negative_envelope,) = axes.plot([], [], zorder=1, **ENVELOPE_STYLE)
    return [real, imaginary, envelope, negative_envelope]


def show_wavefunction(lines: Sequence[Line2D], coordinates: np.ndarray, values: np.ndarray) -> None:
    real, imaginary, envelope, negative_envelope = lines
    real.set_data(coordinates, values.real)
    imaginary.set_data(coordinates, values.imag)
    envelope.set_data(coordinates, np.abs(values))
    negative_envelope.set_data(coordinates, -np.abs(values))


def radio_buttons(
    figure: Figure,
    rect: tuple[float, float, float, float],
    title: str,
    labels: list[str],
    initial: str,
    on_clicked: Callable[[str], None],
) -> RadioButtons:
    axes = figure.add_axes(rect)
    axes.set_title(title, fontsize=10)
    buttons = RadioButtons(axes, labels, active=labels.index(initial))
    buttons.on_clicked(on_clicked)
    return buttons
