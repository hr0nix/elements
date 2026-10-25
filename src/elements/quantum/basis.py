import itertools
from collections.abc import Iterator, Sequence

import numpy as np

from elements.math.utils import frozen, nearly_equal
from elements.quantum.state import StateSpace, StateSpaceVector


def _orthonormal_columns(kets: Sequence[StateSpaceVector]) -> np.ndarray:
    if len(kets) == 0:
        raise ValueError("A basis needs at least one ket")
    state_space = kets[0].state_space
    if not all(ket.is_ket and ket.state_space == state_space for ket in kets):
        raise ValueError("Basis vectors must be kets from the same state space")
    columns = np.hstack([ket.components for ket in kets])
    if not nearly_equal(columns.conj().T @ columns, np.eye(len(kets))):
        raise ValueError("Basis vectors must be orthonormal")
    return columns


class Basis:
    # An orthonormal basis of a state space. matrix holds the basis kets as its columns, written
    # in the label basis of the space, which makes it unitary.
    def __init__(self, kets: Sequence[StateSpaceVector]):
        columns = _orthonormal_columns(kets)
        self.state_space = kets[0].state_space
        if len(kets) != self.state_space.dimension:
            raise ValueError("A basis needs as many kets as its state space has dimensions")
        self._kets = tuple(kets)
        self.matrix = frozen(columns)

    @staticmethod
    def from_columns(state_space: StateSpace, matrix: np.ndarray) -> Basis:
        return Basis([StateSpaceVector(state_space, column[:, np.newaxis]) for column in matrix.T])

    # The kets, which must be orthonormal, followed by kets spanning the rest of their space.
    @staticmethod
    def completing(kets: Sequence[StateSpaceVector]) -> Basis:
        columns = _orthonormal_columns(kets)
        rest = np.linalg.qr(columns, mode="complete").Q[:, len(kets) :]
        return Basis.from_columns(kets[0].state_space, np.hstack([columns, rest]))

    def __len__(self) -> int:
        return len(self._kets)

    def __getitem__(self, index: int) -> StateSpaceVector:
        return self._kets[index]

    def __iter__(self) -> Iterator[StateSpaceVector]:
        return iter(self._kets)

    # Coordinate i is <b_i|ket>, the amplitude of the ket along basis ket i.
    def coordinates(self, ket: StateSpaceVector) -> np.ndarray:
        if not ket.is_ket or ket.state_space != self.state_space:
            raise ValueError("Coordinates are taken of kets from the basis's state space")
        return (self.matrix.conj().T @ ket.components).ravel()

    # The ket sum_i c_i |b_i> with the given coordinates.
    def ket(self, coordinates: np.ndarray) -> StateSpaceVector:
        coordinates = np.asarray(coordinates)
        if coordinates.shape != (len(self),):
            raise ValueError("A ket needs one coordinate per basis ket")
        return StateSpaceVector(self.state_space, self.matrix @ coordinates[:, np.newaxis])

    # Row i, column j is <b_i|o_j>, so column j holds the coordinates of other[j] in this basis.
    def overlaps(self, other: Basis) -> np.ndarray:
        if other.state_space != self.state_space:
            raise ValueError("Overlaps need two bases of the same state space")
        return self.matrix.conj().T @ other.matrix

    # Products of one ket from each basis, in the order of the product space's labels.
    def tensor_product(self, *others: Basis) -> Basis:
        return Basis(
            [first.tensor_product(*rest) for first, *rest in itertools.product(self, *others)]
        )

    def reordered(self, order: Sequence[int]) -> Basis:
        if np.array_equal(order, np.arange(len(self))):
            return self
        return Basis([self._kets[index] for index in order])

    def __repr__(self) -> str:
        return f"Basis({', '.join(repr(ket) for ket in self)})"
