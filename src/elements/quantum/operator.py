from collections.abc import Sequence
from numbers import Number

import numpy as np

from elements.math.typechecks import is_complex
from elements.math.utils import DEFAULT_TOLERANCE, frozen
from elements.quantum.state import StateSpace, StateSpaceVector


class Operator:
    # eigenvalues and is_normal are trusted as caches: pass them only when known exactly.
    def __init__(
        self,
        state_space: StateSpace,
        matrix: np.ndarray,
        *,
        eigenvalues: np.ndarray | None = None,
        is_normal: bool | None = None,
    ):
        dimension = state_space.dimension
        if not is_complex(matrix) or matrix.shape != (dimension, dimension):
            raise ValueError("Matrix must be complex and square, matching the state space")

        self.state_space = state_space
        self.matrix = frozen(matrix)
        self._eigenvalues = None if eigenvalues is None else frozen(eigenvalues)
        self._is_normal = is_normal

    @staticmethod
    def identity(state_space: StateSpace) -> Operator:
        dimension = state_space.dimension
        return Operator(
            state_space,
            np.eye(dimension, dtype=complex),
            eigenvalues=np.ones(dimension, dtype=complex),
            is_normal=True,
        )

    @staticmethod
    def spectral(eigenpairs: Sequence[tuple[complex, StateSpaceVector]]) -> Operator:
        if len(eigenpairs) == 0:
            raise ValueError("Spectral decomposition requires at least one eigenpair")
        state_space = eigenpairs[0][1].state_space
        if not all(vector.is_ket and vector.state_space == state_space for _, vector in eigenpairs):
            raise ValueError("Eigenvectors must be kets from the same state space")

        eigenvalues = np.array([value for value, _ in eigenpairs], dtype=complex)
        eigenvectors = np.hstack([vector.components for _, vector in eigenpairs])
        if not np.allclose(eigenvectors.conj().T @ eigenvectors, np.eye(len(eigenpairs))):
            raise ValueError("Eigenvectors must be orthonormal")

        matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        # Directions not spanned by the eigenvectors get eigenvalue 0.
        missing = state_space.dimension - len(eigenpairs)
        eigenvalues = np.concatenate([eigenvalues, np.zeros(missing, dtype=complex)])
        return Operator(state_space, matrix, eigenvalues=eigenvalues, is_normal=True)

    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            self._eigenvalues = frozen(np.linalg.eigvals(self.matrix))
        return self._eigenvalues

    @property
    def is_normal(self) -> bool:
        if self._is_normal is None:
            adjoint = self.matrix.conj().T
            self._is_normal = np.allclose(self.matrix @ adjoint, adjoint @ self.matrix)
        return self._is_normal

    # Eigenvalues decide these only for normal matrices: [[1, 1], [0, 1]] has eigenvalues {1, 1}.
    @property
    def is_hermitian(self) -> bool:
        return self.is_normal and np.allclose(self.eigenvalues.imag, 0.0)

    @property
    def is_unitary(self) -> bool:
        return self.is_normal and np.allclose(np.abs(self.eigenvalues), 1.0)

    def kronecker_product(self, other: Operator) -> Operator:
        return Operator(
            self.state_space.tensor_product(other.state_space),
            np.kron(self.matrix, other.matrix),
        )

    @property
    def dagger(self) -> Operator:
        return Operator(self.state_space, self.matrix.conj().T)

    def __mul__(self, other: complex) -> Operator:
        if not isinstance(other, Number):
            return NotImplemented
        return Operator(self.state_space, self.matrix * other)

    def __rmul__(self, other: complex) -> Operator:
        return self * other

    def __truediv__(self, other: complex) -> Operator:
        return self * (1 / other)

    def __neg__(self) -> Operator:
        return (-1) * self

    def __add__(self, other: Operator) -> Operator:
        if not isinstance(other, Operator):
            return NotImplemented
        if other.state_space != self.state_space:
            raise ValueError("Operators must act on the same state space")
        return Operator(self.state_space, self.matrix + other.matrix)

    def __sub__(self, other: Operator) -> Operator:
        return self + (-other)

    def __matmul__(self, other: Operator | StateSpaceVector) -> Operator | StateSpaceVector:
        if isinstance(other, Operator):
            if other.state_space != self.state_space:
                raise ValueError("Operators must act on the same state space")
            return Operator(self.state_space, self.matrix @ other.matrix)

        if isinstance(other, StateSpaceVector):
            if not other.is_ket:
                raise ValueError("Operators act on kets from the left")
            if other.state_space != self.state_space:
                raise ValueError("Ket must be from the operator's state space")
            return StateSpaceVector(self.state_space, self.matrix @ other.components)

        return NotImplemented

    def __rmatmul__(self, other: StateSpaceVector) -> StateSpaceVector:
        if not isinstance(other, StateSpaceVector):
            return NotImplemented
        if not other.is_bra:
            raise ValueError("Operators act on bras from the right")
        if other.state_space != self.state_space:
            raise ValueError("Bra must be from the operator's state space")
        return StateSpaceVector(self.state_space, other.components @ self.matrix)

    def close_to(self, other: Operator, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        if self.state_space != other.state_space:
            return False
        return np.allclose(self.matrix, other.matrix, rtol=0.0, atol=tolerance)

    def __repr__(self) -> str:
        return f"Operator on {self.state_space}\n{self.matrix}"
