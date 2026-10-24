from abc import ABC, abstractmethod

import numpy as np

from elements.quantum.continuous_approx import CyclicalGrid1DStateSpace, Operators1D
from elements.quantum.operator import Operator
from elements.util.cache import cached


class Task(ABC):
    name: str

    @abstractmethod
    def potential(self, x: np.ndarray, mass: float) -> np.ndarray: ...

    # Cached here rather than by Operators1D.hamiltonian, which sees a fresh potential closure
    # on every call. Diagonalizing a new Hamiltonian is the slowest step of a restart.
    @cached
    def hamiltonian(self, state: CyclicalGrid1DStateSpace, mass: float) -> Operator:
        return Operators1D.hamiltonian(state, mass, lambda x: self.potential(x, mass))

    # Position and momentum at the given time of a classical particle that started at (x0, p0).
    # Positions are on the line, not wrapped round the circle.
    @abstractmethod
    def classical(self, x0: float, p0: float, mass: float, time: float) -> tuple[float, float]: ...


class FreeParticle(Task):
    name = "Free particle"

    def potential(self, x: np.ndarray, mass: float) -> np.ndarray:
        return np.zeros_like(x)

    # The kinetic energy on its own keeps its exact plane-wave eigenbasis, so nothing has to
    # be diagonalized numerically.
    def hamiltonian(self, state: CyclicalGrid1DStateSpace, mass: float) -> Operator:
        return Operators1D.kinetic_energy(state, mass)

    def classical(self, x0: float, p0: float, mass: float, time: float) -> tuple[float, float]:
        return x0 + p0 * time / mass, p0


# A fixed spring rather than a fixed frequency: heavier particles then oscillate more slowly
# instead of reaching momenta beyond what the grid resolves. With mass 1 the default spring
# gives omega = 1/2, whose coherent states, the packets that keep their shape, have width 1.
class HarmonicOscillator(Task):
    name = "Harmonic oscillator"

    def __init__(self, spring_constant: float = 0.25) -> None:
        self.spring_constant = spring_constant

    def potential(self, x: np.ndarray, mass: float) -> np.ndarray:
        return 0.5 * self.spring_constant * x**2

    def classical(self, x0: float, p0: float, mass: float, time: float) -> tuple[float, float]:
        omega = np.sqrt(self.spring_constant / mass)
        phase = omega * time
        x = x0 * np.cos(phase) + p0 / (mass * omega) * np.sin(phase)
        p = p0 * np.cos(phase) - mass * omega * x0 * np.sin(phase)
        return float(x), float(p)


# A heavy object at x = 0. On a line, Gauss's law makes its pull the same strength at every
# distance, so the potential grows linearly away from the object.
class Gravity(Task):
    name = "Gravity"

    def __init__(self, g: float = 0.5) -> None:
        self.g = g

    def potential(self, x: np.ndarray, mass: float) -> np.ndarray:
        return mass * self.g * np.abs(x)

    # The particle falls through the object and out the other side: a parabolic arc on each
    # side of x = 0, each lasting 2u / g, where u is the speed at x = 0.
    def classical(self, x0: float, p0: float, mass: float, time: float) -> tuple[float, float]:
        velocity = p0 / mass
        speed_at_centre = np.sqrt(velocity**2 + 2 * self.g * abs(x0))
        if speed_at_centre == 0:
            return 0.0, 0.0
        arc = 2 * speed_at_centre / self.g
        # Time since the particle last left x = 0 heading right.
        if x0 >= 0:
            elapsed = (speed_at_centre - velocity) / self.g
        else:
            elapsed = arc + (speed_at_centre + velocity) / self.g
        elapsed = (elapsed + time) % (2 * arc)
        side = 1.0
        if elapsed >= arc:
            side, elapsed = -1.0, elapsed - arc
        x = side * (speed_at_centre * elapsed - 0.5 * self.g * elapsed**2)
        return float(x), float(side * mass * (speed_at_centre - self.g * elapsed))
