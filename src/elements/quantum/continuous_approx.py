from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from elements.math.utils import frozen, nearly_equal
from elements.quantum.basis import Basis
from elements.quantum.operator import Operator
from elements.quantum.state import StateSpace, StateSpaceVector
from elements.util.cache import cached


class CyclicalGrid1DStateSpace(StateSpace):
    # A circle of circumference max_x - min_x: max_x is the same point as min_x,
    # so it is not a tick.
    def __init__(self, min_x: float, max_x: float, num_points: int) -> None:
        if max_x <= min_x:
            raise ValueError("Grid needs max_x > min_x")
        if num_points < 1 or num_points % 2 == 0:
            # With an even count, the zigzag (-1)^k is the plane wave for both +pi/dx and
            # -pi/dx, so it would have no definite momentum.
            raise ValueError("Number of points must be positive and odd")
        self._min_x = min_x
        self._max_x = max_x
        self.ticks = frozen(np.linspace(min_x, max_x, num_points, endpoint=False))
        # Enough decimals that neighbouring ticks never round to the same label.
        decimals = max(2, int(np.ceil(-np.log10(self.dx))) + 1)
        labels = [f"{tick:.{decimals}f}" for tick in self.ticks]
        super().__init__(*labels)

    @property
    def min_x(self) -> float:
        return self._min_x

    @property
    def max_x(self) -> float:
        return self._max_x

    @property
    def length(self) -> float:
        return self.max_x - self.min_x

    # The point opposite the seam.
    @property
    def middle(self) -> float:
        return self.min_x + self.length / 2

    @property
    def num_points(self) -> int:
        return len(self.ticks)

    @property
    def dx(self) -> float:
        return self.length / self.num_points

    def __repr__(self) -> str:
        return f"CyclicalGrid1DStateSpace(min_x={self.min_x}, max_x={self.max_x}, dx={self.dx})"


class ContinuousBasis:
    # The eigenbasis of a Hermitian operator whose eigenvalues sample a continuum at even
    # spacing, such as position or momentum on the grid. Kets |x> of a continuous basis are
    # normalized to a delta function, <x|x'> = delta(x - x'), which on the grid makes
    # |x> = |x_i> / sqrt(dx), with dx the spacing between eigenvalues.
    def __init__(self, operator: Operator):
        if not operator.is_hermitian:
            raise ValueError("A continuous basis is the eigenbasis of a Hermitian operator")
        spacings = np.diff(operator.eigenvalues)
        if spacings[0] <= 0 or not nearly_equal(spacings, spacings[0]):
            raise ValueError("Eigenvalues must be distinct and evenly spaced")
        self.points = operator.eigenvalues
        self.spacing = float(spacings[0])
        self.basis = operator.eigenbasis

    # psi(x) = <x|psi> at each point x.
    def wavefunction(self, ket: StateSpaceVector) -> np.ndarray:
        return self.basis.coordinates(ket) / np.sqrt(self.spacing)

    # Column j is the wavefunction of kets[j].
    def wavefunctions(self, kets: Basis) -> np.ndarray:
        return self.basis.overlaps(kets) / np.sqrt(self.spacing)

    # |psi> = integral of psi(x) |x> dx, which on the grid is a sum over the points.
    def ket(self, wavefunction: np.ndarray) -> StateSpaceVector:
        return self.basis.ket(np.asarray(wavefunction) * np.sqrt(self.spacing))


class Operators1D:
    @staticmethod
    @cached
    def position(grid: CyclicalGrid1DStateSpace) -> Operator:
        return Operator.diagonal(grid.basis, grid.ticks)

    # Diagonal in the plane waves that fit on the circle, whose momenta (hbar = 1) are the
    # multiples of 2 pi / length up to the grid's resolution.
    @staticmethod
    @cached
    def momentum(grid: CyclicalGrid1DStateSpace) -> Operator:
        quanta = np.arange(grid.num_points) - grid.num_points // 2
        momenta = 2 * np.pi / grid.length * quanta
        plane_waves = Basis([States1D.plane_wave(grid, p) for p in momenta])
        return Operator.diagonal(plane_waves, momenta)

    @staticmethod
    @cached
    def momentum_squared(grid: CyclicalGrid1DStateSpace) -> Operator:
        return Operators1D.momentum(grid).map_eigenvalues(np.square)

    @staticmethod
    @cached
    def kinetic_energy(grid: CyclicalGrid1DStateSpace, mass: float) -> Operator:
        return Operators1D.momentum(grid).map_eigenvalues(lambda p: p**2 / (2 * mass))

    # V(X). Not cached: every call brings its own potential, often a fresh closure.
    @staticmethod
    def potential_energy(
        grid: CyclicalGrid1DStateSpace, v: Callable[[np.ndarray], np.ndarray]
    ) -> Operator:
        return Operators1D.position(grid).map_eigenvalues(v)

    @staticmethod
    def hamiltonian(
        grid: CyclicalGrid1DStateSpace, mass: float, v: Callable[[np.ndarray], np.ndarray]
    ) -> Operator:
        return Operators1D.kinetic_energy(grid, mass) + Operators1D.potential_energy(grid, v)

    # Moves a state round the circle by any distance, not only whole ticks.
    @staticmethod
    @cached
    def translation(grid: CyclicalGrid1DStateSpace, distance: float) -> Operator:
        return Operators1D.momentum(grid).unitary(distance)

    # Raises the momentum of a state. Position generates shifts of momentum the opposite way
    # to how momentum generates shifts of position, as [P, X] = -[X, P]. Unless momentum is a
    # multiple of 2 pi / length, the phase jumps at the seam, so only boost states that vanish
    # there.
    @staticmethod
    @cached
    def boost(grid: CyclicalGrid1DStateSpace, momentum: float) -> Operator:
        return Operators1D.position(grid).unitary(-momentum)


class States1D:
    # exp(i momentum x) / sqrt(length): spread evenly over the circle, with total probability 1.
    @staticmethod
    def plane_wave(grid: CyclicalGrid1DStateSpace, momentum: float) -> StateSpaceVector:
        position = ContinuousBasis(Operators1D.position(grid))
        return position.ket(np.exp(1j * momentum * position.points) / np.sqrt(grid.length))

    @staticmethod
    @cached
    def gaussian_at_rest(grid: CyclicalGrid1DStateSpace, sigma: float) -> StateSpaceVector:
        if sigma <= 0:
            raise ValueError("Packet width must be positive")
        position = ContinuousBasis(Operators1D.position(grid))
        x = position.points - grid.middle
        wavefunction = (2 * np.pi * sigma**2) ** -0.25 * np.exp(-(x**2) / (4 * sigma**2))
        # The circle cuts off the tails of a packet nearly as wide as itself, which leaves the
        # norm short of 1.
        return position.ket(wavefunction).normalized

    @staticmethod
    @cached
    def packet(
        grid: CyclicalGrid1DStateSpace, x0: float, p0: float, sigma: float
    ) -> StateSpaceVector:
        # Boost opposite the seam, where the packet at rest vanishes, then move it into place.
        moving = Operators1D.boost(grid, p0) @ States1D.gaussian_at_rest(grid, sigma)
        return Operators1D.translation(grid, x0 - grid.middle) @ moving


# A particle of the given mass in a potential on the line: solved quantum mechanically on a
# grid, and classically. Tasks are values, so equal tasks share a cached Hamiltonian.
@dataclass(frozen=True, kw_only=True)
class Task1D(ABC):
    name: ClassVar[str]
    mass: float = 1.0

    @abstractmethod
    def potential(self, x: np.ndarray) -> np.ndarray: ...

    # Cached here rather than by Operators1D.hamiltonian, which sees the potential only as a
    # function and so cannot tell when two tasks are equal. Diagonalizing a new Hamiltonian is
    # the slowest step of a restart.
    @cached
    def hamiltonian(self, grid: CyclicalGrid1DStateSpace) -> Operator:
        return Operators1D.hamiltonian(grid, self.mass, self.potential)

    # H(x, p) = p^2 / 2m + V(x) of the classical particle.
    def classical_energy(self, x: float, p: float) -> float:
        return p**2 / (2 * self.mass) + float(self.potential(np.asarray(x)))

    # Position and momentum, together the classical state, at the given time of a particle that
    # started at (x0, p0). Positions are on the line, not wrapped round the circle.
    @abstractmethod
    def classical_state(self, x0: float, p0: float, time: float) -> tuple[float, float]: ...


@dataclass(frozen=True, kw_only=True)
class FreeParticle(Task1D):
    name = "Free particle"

    def potential(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def classical_state(self, x0: float, p0: float, time: float) -> tuple[float, float]:
        return x0 + p0 * time / self.mass, p0


# A fixed spring rather than a fixed frequency: heavier particles then oscillate more slowly
# instead of reaching momenta beyond what the grid resolves. With mass 1 the default spring
# gives omega = 1/2, whose coherent states, the packets that keep their shape, have width 1.
@dataclass(frozen=True, kw_only=True)
class HarmonicOscillator(Task1D):
    name = "Harmonic oscillator"
    spring_constant: float = 0.25

    @property
    def omega(self) -> float:
        return float(np.sqrt(self.spring_constant / self.mass))

    def potential(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * self.spring_constant * x**2

    def classical_state(self, x0: float, p0: float, time: float) -> tuple[float, float]:
        phase = self.omega * time
        x = x0 * np.cos(phase) + p0 / (self.mass * self.omega) * np.sin(phase)
        p = p0 * np.cos(phase) - self.mass * self.omega * x0 * np.sin(phase)
        return float(x), float(p)


# A heavy object at x = 0. On a line, Gauss's law makes its pull the same strength at every
# distance, so the potential grows linearly away from the object.
@dataclass(frozen=True, kw_only=True)
class Gravity(Task1D):
    name = "Gravity"
    g: float = 0.5

    def potential(self, x: np.ndarray) -> np.ndarray:
        return self.mass * self.g * np.abs(x)

    # The particle falls through the object and out the other side. Its velocity falls at rate
    # g while x > 0 and rises at rate g while x < 0, so with tau the time since it last crossed
    # x = 0 heading right, wrapped into one period [-2u / g, 2u / g), and u its speed at x = 0:
    # v = u - g |tau| and x = tau (u - g |tau| / 2), a parabolic arc on each side.
    def classical_state(self, x0: float, p0: float, time: float) -> tuple[float, float]:
        v0 = p0 / self.mass
        u = np.sqrt(v0**2 + 2 * self.g * abs(x0))
        if u == 0:
            # At rest at the bottom, where it stays.
            return 0.0, 0.0
        period = 4 * u / self.g
        tau0 = np.copysign((u - v0) / self.g, x0)
        tau = (tau0 + time + period / 2) % period - period / 2
        x = tau * (u - self.g * abs(tau) / 2)
        return float(x), float(self.mass * (u - self.g * abs(tau)))
