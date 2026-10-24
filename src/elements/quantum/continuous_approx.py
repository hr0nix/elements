from collections.abc import Callable

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

    # The ket whose component along each position eigenvector |x>, a basis ket of the grid,
    # is f(x).
    def sample(self, f: Callable[[np.ndarray], np.ndarray]) -> StateSpaceVector:
        return self.basis.ket(f(self.ticks))

    def __repr__(self) -> str:
        return f"CyclicalGrid1DStateSpace(min_x={self.min_x}, max_x={self.max_x}, dx={self.dx})"


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
        plane_waves = Basis([WaveFunctions1D.plane_wave(grid, p) for p in momenta])
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

    # exp(-i distance P): moves a state round the circle by any distance, not only whole ticks.
    @staticmethod
    @cached
    def translation(grid: CyclicalGrid1DStateSpace, distance: float) -> Operator:
        return (-1j * distance * Operators1D.momentum(grid)).exp()

    # exp(i momentum X). Unless momentum is a multiple of 2 pi / length, the phase jumps at the
    # seam, so only boost states that vanish there.
    @staticmethod
    @cached
    def boost(grid: CyclicalGrid1DStateSpace, momentum: float) -> Operator:
        return (1j * momentum * Operators1D.position(grid)).exp()


class WaveFunctions1D:
    @staticmethod
    def plane_wave(grid: CyclicalGrid1DStateSpace, momentum: float) -> StateSpaceVector:
        return grid.sample(lambda x: np.exp(1j * momentum * x)).normalized

    @staticmethod
    @cached
    def gaussian_at_rest(grid: CyclicalGrid1DStateSpace, sigma: float) -> StateSpaceVector:
        if sigma <= 0:
            raise ValueError("Packet width must be positive")
        return grid.sample(lambda x: np.exp(-((x - grid.middle) ** 2) / (4 * sigma**2))).normalized

    @staticmethod
    @cached
    def packet(
        grid: CyclicalGrid1DStateSpace, x0: float, p0: float, sigma: float
    ) -> StateSpaceVector:
        # Boost opposite the seam, where the packet at rest vanishes, then move it into place.
        moving = Operators1D.boost(grid, p0) @ WaveFunctions1D.gaussian_at_rest(grid, sigma)
        return Operators1D.translation(grid, x0 - grid.middle) @ moving

    @staticmethod
    def position_wavefunction(vector: StateSpaceVector) -> tuple[np.ndarray, np.ndarray]:
        return WaveFunctions1D.wavefunction(Operators1D.position(vector.state_space), vector)

    @staticmethod
    def momentum_wavefunction(vector: StateSpaceVector) -> tuple[np.ndarray, np.ndarray]:
        return WaveFunctions1D.wavefunction(Operators1D.momentum(vector.state_space), vector)

    # The eigenvalues of a Hermitian operator that samples a continuum, and the vector's
    # amplitude density over them.
    @staticmethod
    def wavefunction(operator: Operator, vector: StateSpaceVector) -> tuple[np.ndarray, np.ndarray]:
        if not operator.is_hermitian:
            raise ValueError("Wavefunctions are read off the eigenbasis of a Hermitian operator")
        eigenvalues = operator.eigenvalues
        amplitudes = operator.eigenbasis.coordinates(vector)
        return eigenvalues, WaveFunctions1D.amplitude_density(eigenvalues, amplitudes)

    # Each amplitude along an eigenvector is the amplitude for a cell as wide as the spacing
    # between eigenvalues, so dividing by the square root of that spacing turns amplitudes into
    # an amplitude density. Amplitudes run along the first axis, one per eigenvalue.
    @staticmethod
    def amplitude_density(eigenvalues: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        spacings = np.diff(eigenvalues)
        if spacings[0] <= 0 or not nearly_equal(spacings, spacings[0]):
            raise ValueError("Eigenvalues must be distinct and evenly spaced")
        return amplitudes / np.sqrt(spacings[0])
