from collections.abc import Callable

import numpy as np

from elements.math.utils import frozen
from elements.quantum.operator import Operator
from elements.quantum.state import StateSpace, StateSpaceVector


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
        super().__init__(labels)

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
        return f"CyclicalGridState(min_x={self.min_x}, max_x={self.max_x}, dx={self.dx})"


class Operators1D:
    # Momenta of the plane waves that fit on the circle (hbar = 1), in FFT order:
    # zero, then positive, then negative.
    @staticmethod
    def _momentum_eigenvalues(state: CyclicalGrid1DStateSpace) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(state.num_points, d=state.dx)

    @staticmethod
    def _momentum_eigenvectors(state: CyclicalGrid1DStateSpace) -> np.ndarray:
        # Column j is the plane wave exp(i p_j x) / sqrt(n) sampled at the ticks,
        # with p_j = _momentum_eigenvalues(state)[j].
        momenta = Operators1D._momentum_eigenvalues(state)
        return np.exp(1j * np.outer(state.ticks, momenta)) / np.sqrt(state.num_points)

    @staticmethod
    def _momentum_basis_with_eigenvalues(
        state: CyclicalGrid1DStateSpace, eigenvalues: np.ndarray
    ) -> Operator:
        eigenvectors = Operators1D._momentum_eigenvectors(state)
        return Operator(
            state,
            (eigenvectors * eigenvalues) @ eigenvectors.conj().T,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            is_normal=True,
        )

    @staticmethod
    def position(state: CyclicalGrid1DStateSpace) -> Operator:
        return Operator.diagonal(state, state.ticks)

    @staticmethod
    def potential_energy(
        state: CyclicalGrid1DStateSpace, v: Callable[[np.ndarray], np.ndarray]
    ) -> Operator:
        values = v(state.ticks)
        if len(values) != state.num_points:
            raise ValueError("Values must have the same dimension as the state space")
        return Operator.diagonal(state, values)

    @staticmethod
    def momentum(state: CyclicalGrid1DStateSpace) -> Operator:
        momenta = Operators1D._momentum_eigenvalues(state)
        return Operators1D._momentum_basis_with_eigenvalues(state, momenta)

    @staticmethod
    def momentum_squared(state: CyclicalGrid1DStateSpace) -> Operator:
        momenta = Operators1D._momentum_eigenvalues(state)
        return Operators1D._momentum_basis_with_eigenvalues(state, momenta**2)

    @staticmethod
    def kinetic_energy(state: CyclicalGrid1DStateSpace, mass: float) -> Operator:
        momenta = Operators1D._momentum_eigenvalues(state)
        return Operators1D._momentum_basis_with_eigenvalues(state, momenta**2 / (2 * mass))

    @staticmethod
    def hamiltonian(
        state: CyclicalGrid1DStateSpace, mass: float, v: Callable[[np.ndarray], np.ndarray]
    ) -> Operator:
        return Operators1D.kinetic_energy(state, mass) + Operators1D.potential_energy(state, v)

    # exp(-i distance P): moves a state round the circle by any distance, not only whole ticks.
    @staticmethod
    def translation(state: CyclicalGrid1DStateSpace, distance: float) -> Operator:
        momenta = Operators1D._momentum_eigenvalues(state)
        return Operators1D._momentum_basis_with_eigenvalues(state, np.exp(-1j * momenta * distance))

    # exp(i momentum X). Unless momentum is a multiple of 2 pi / length, the phase jumps at the
    # seam, so only boost states that vanish there.
    @staticmethod
    def boost(state: CyclicalGrid1DStateSpace, momentum: float) -> Operator:
        return Operator.diagonal(state, np.exp(1j * momentum * state.ticks))


class WaveFunctions1D:
    @staticmethod
    def gaussian_at_rest(state: CyclicalGrid1DStateSpace, sigma: float) -> StateSpaceVector:
        if sigma <= 0:
            raise ValueError("Packet width must be positive")
        values = np.exp(-((state.ticks - state.middle) ** 2) / (4 * sigma**2))
        return StateSpaceVector(state, values[:, np.newaxis]).normalized

    @staticmethod
    def packet(
        state: CyclicalGrid1DStateSpace, x0: float, p0: float, sigma: float
    ) -> StateSpaceVector:
        # Boost opposite the seam, where the packet at rest vanishes, then move it into place.
        moving = Operators1D.boost(state, p0) @ WaveFunctions1D.gaussian_at_rest(state, sigma)
        return Operators1D.translation(state, x0 - state.middle) @ moving
