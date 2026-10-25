import functools
from collections.abc import Callable, Sequence
from numbers import Number

import numpy as np

from elements.math.utils import DEFAULT_TOLERANCE, frozen, nearly_equal
from elements.quantum.basis import Basis
from elements.quantum.state import StateSpace, StateSpaceVector


class Operator:
    # An operator is given by its matrix or by its eigenvalues on an orthonormal eigenbasis,
    # and works out the other on first use. Operators given by an eigenbasis, such as functions
    # of other operators, act on kets through it and never need a matrix unless they meet
    # another operator. Build operators with from_matrix, diagonal or the constructors built on
    # them rather than calling __init__ directly.
    def __init__(
        self,
        state_space: StateSpace,
        *,
        matrix: np.ndarray | None = None,
        eigenvalues: np.ndarray | None = None,
        eigenbasis: Basis | None = None,
    ):
        if (eigenvalues is None) != (eigenbasis is None):
            raise ValueError("Eigenvalues come together with the eigenbasis they belong to")
        has_spectrum = eigenbasis is not None
        if matrix is not None and has_spectrum:
            raise ValueError("An operator is given by a matrix or by its spectrum, not both")
        if matrix is None and not has_spectrum:
            raise ValueError("An operator needs a matrix or its spectrum")

        self.state_space = state_space
        self._matrix: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenbasis: Basis | None = None
        self._is_normal: bool | None = None
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=complex)
            if matrix.shape != (state_space.dimension, state_space.dimension):
                raise ValueError("Matrix must be square, matching the state space")
            self._matrix = frozen(matrix)
        else:
            if eigenbasis.state_space != state_space:
                raise ValueError("Eigenbasis must be a basis of the operator's state space")
            if np.shape(eigenvalues) != (len(eigenbasis),):
                raise ValueError("Need one eigenvalue per eigenbasis ket")
            self._remember_eigenbasis(eigenvalues, eigenbasis)

    @staticmethod
    def from_matrix(state_space: StateSpace, matrix: np.ndarray) -> Operator:
        return Operator(state_space, matrix=matrix)

    # sum_i values_i |b_i><b_i|: the operator that is diagonal in the basis, with the given values
    # along its diagonal.
    @staticmethod
    def diagonal(basis: Basis, values: np.ndarray) -> Operator:
        return Operator(basis.state_space, eigenvalues=values, eigenbasis=basis)

    # sum_i lambda_i |v_i><v_i| for an operator given by its eigenbasis.
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            basis = self._eigenbasis
            self._matrix = frozen((basis.matrix * self._eigenvalues) @ basis.matrix.conj().T)
        return self._matrix

    @staticmethod
    def identity(state_space: StateSpace) -> Operator:
        return Operator.diagonal(state_space.basis, np.ones(state_space.dimension))

    @staticmethod
    def spectral(eigenpairs: Sequence[tuple[complex, StateSpaceVector]]) -> Operator:
        if len(eigenpairs) == 0:
            raise ValueError("Spectral decomposition requires at least one eigenpair")
        basis = Basis.completing([vector for _, vector in eigenpairs])
        # Directions not spanned by the eigenvectors get eigenvalue 0.
        missing = len(basis) - len(eigenpairs)
        return Operator.diagonal(basis, [value for value, _ in eigenpairs] + [0] * missing)

    # |psi><psi|, the density of the pure state psi: eigenvalue 1 on psi, 0 orthogonal to it.
    @staticmethod
    def density(state: StateSpaceVector) -> Operator:
        if not (state.is_ket and state.is_normalized):
            raise ValueError("A density is built from a normalized ket")
        return Operator.spectral([(1, state)])

    # Hermitian operators keep their eigenpairs in order of increasing eigenvalue, with the
    # eigenvalues stored as real numbers.
    def _remember_eigenbasis(self, values: np.ndarray, basis: Basis) -> None:
        values = np.asarray(values, dtype=complex)
        if not values.imag.any():
            order = np.argsort(values.real, kind="stable")
            values, basis = values.real[order], basis.reordered(order)
        self._eigenvalues = frozen(values)
        self._eigenbasis = basis
        self._is_normal = True

    def _compute_eigenbasis(self) -> None:
        if not self.is_normal:
            raise ValueError("Only normal operators have an orthonormal basis of eigenvectors")
        if self.close_to(self.dagger):
            # A complex solver returns each eigenvector times an arbitrary phase, even when the
            # matrix is real and a real eigenvector exists, as for any real potential.
            real = nearly_equal(self.matrix, self.matrix.real)
            values, vectors = np.linalg.eigh(self.matrix.real if real else self.matrix)
        else:
            values, vectors = np.linalg.eig(self.matrix)
            # A normal operator's eigenspaces are mutually orthogonal,
            # so orthonormalizing keeps every vector inside its own eigenspace.
            vectors = np.linalg.qr(vectors).Q
        self._remember_eigenbasis(values, Basis.from_columns(self.state_space, vectors))

    # Real and increasing for Hermitian operators, in no particular order otherwise.
    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            if self.is_normal:
                self._compute_eigenbasis()
            else:
                self._eigenvalues = frozen(np.linalg.eigvals(self.matrix))
        return self._eigenvalues

    # Ordered like the eigenvalues.
    @property
    def eigenbasis(self) -> Basis:
        if self._eigenbasis is None:
            self._compute_eigenbasis()
        return self._eigenbasis

    # f(A) = sum_i f(lambda_i) |v_i><v_i|: the eigenvectors stay, each eigenvalue goes through f.
    def map_eigenvalues(self, f: Callable[[np.ndarray], np.ndarray]) -> Operator:
        return Operator.diagonal(self.eigenbasis, f(self.eigenvalues))

    def exp(self) -> Operator:
        return self.map_eigenvalues(np.exp)

    # exp(-i parameter G), the unitary generated by a Hermitian G: H generates evolution in
    # time, P translation in space, and a spin component rotation about its axis.
    def unitary(self, parameter: float) -> Operator:
        if not self.is_hermitian:
            raise ValueError("Only Hermitian operators generate unitaries")
        return (-1j * parameter * self).exp()

    @property
    def is_normal(self) -> bool:
        if self._is_normal is None:
            adjoint = self.dagger
            self._is_normal = (self @ adjoint).close_to(adjoint @ self)
        return self._is_normal

    # Eigenvalues decide these only for normal matrices: [[1, 1], [0, 1]] has eigenvalues {1, 1}.
    @property
    def is_hermitian(self) -> bool:
        return self.is_normal and nearly_equal(self.eigenvalues, self.eigenvalues.real)

    @property
    def is_unitary(self) -> bool:
        return self.is_normal and nearly_equal(np.abs(self.eigenvalues), 1.0)

    # Its eigenvalues are the probabilities of a mixture of its eigenvectors.
    @property
    def is_density(self) -> bool:
        return (
            self.is_hermitian
            and nearly_equal(self.trace(), 1.0)
            and bool(np.all(self.eigenvalues >= -DEFAULT_TOLERANCE))
        )

    # Products of eigenvectors are eigenvectors of the product, with the eigenvalues multiplied.
    # Tensor products build small spaces, so diagonalizing a factor to get there is cheap.
    def tensor_product(self, *others: Operator) -> Operator:
        operators = (self, *others)
        if all(operator.is_normal for operator in operators):
            basis = self.eigenbasis.tensor_product(*(other.eigenbasis for other in others))
            eigenvalues = functools.reduce(
                np.kron, [operator.eigenvalues for operator in operators]
            )
            return Operator.diagonal(basis, eigenvalues)
        return Operator.from_matrix(
            self.state_space.tensor_product(*(other.state_space for other in others)),
            functools.reduce(np.kron, [operator.matrix for operator in operators]),
        )

    # A normal operator's adjoint has the same eigenvectors, with conjugated eigenvalues. Only an
    # eigenbasis already known is used: is_normal and the eigensolver take the adjoint
    # themselves, and scaling must never set off a diagonalization.
    @property
    def dagger(self) -> Operator:
        if self._eigenbasis is not None:
            return Operator.diagonal(self._eigenbasis, self._eigenvalues.conj())
        return Operator.from_matrix(self.state_space, self.matrix.conj().T)

    # Scaling keeps the eigenvectors and scales the eigenvalues, when they are already known.
    def __mul__(self, other: complex) -> Operator:
        if not isinstance(other, Number):
            return NotImplemented
        if self._eigenbasis is not None:
            return Operator.diagonal(self._eigenbasis, self._eigenvalues * other)
        return Operator.from_matrix(self.state_space, self.matrix * other)

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
        return Operator.from_matrix(self.state_space, self.matrix + other.matrix)

    def __sub__(self, other: Operator) -> Operator:
        return self + (-other)

    def __matmul__(self, other: Operator | StateSpaceVector) -> Operator | StateSpaceVector:
        if isinstance(other, Operator):
            if other.state_space != self.state_space:
                raise ValueError("Operators must act on the same state space")
            return Operator.from_matrix(self.state_space, self.matrix @ other.matrix)

        if isinstance(other, StateSpaceVector):
            if not other.is_ket:
                raise ValueError("Operators act on kets from the left")
            if other.state_space != self.state_space:
                raise ValueError("Ket must be from the operator's state space")
            if self._matrix is None:
                # sum_i lambda_i |v_i><v_i|psi>: each coordinate in the eigenbasis is scaled by
                # its eigenvalue.
                basis = self._eigenbasis
                return basis.ket(self._eigenvalues * basis.coordinates(other))
            return StateSpaceVector(self.state_space, self.matrix @ other.components)

        return NotImplemented

    # <phi|A is the bra of A^dagger |phi>.
    def __rmatmul__(self, other: StateSpaceVector) -> StateSpaceVector:
        if not isinstance(other, StateSpaceVector):
            return NotImplemented
        if not other.is_bra:
            raise ValueError("Operators act on bras from the right")
        if other.state_space != self.state_space:
            raise ValueError("Bra must be from the operator's state space")
        return (self.dagger @ other.dagger).dagger

    def trace(self) -> complex:
        return complex(np.trace(self.matrix))

    # tr(rho^2): 1 for a pure state, down to 1 / dimension for the most mixed one.
    def purity(self) -> float:
        if not self.is_density:
            raise ValueError("Purity is defined for densities")
        return (self @ self).trace().real

    def partial_trace(self, keep: int | Sequence[int]) -> Operator:
        factors = self.state_space.factors
        kept = {keep} if isinstance(keep, int) else set(keep)
        if not kept <= set(range(len(factors))):
            raise ValueError(f"Factor indices must be between 0 and {len(factors) - 1}")
        if not kept:
            raise ValueError("Partial trace must keep at least one factor; use trace() instead")

        # Axes are (row of each factor..., column of each factor...). Tracing from the last
        # factor down leaves the row axes of the earlier factors where they were.
        tensor = self.matrix.reshape([factor.dimension for factor in factors] * 2)
        for index in reversed(range(len(factors))):
            if index not in kept:
                tensor = np.trace(tensor, axis1=index, axis2=index + tensor.ndim // 2)
        # Kept factors stay in their original order, whatever order keep lists them in.
        first, *rest = (factors[index] for index in sorted(kept))
        space = first.tensor_product(*rest)
        return Operator.from_matrix(space, tensor.reshape(space.dimension, space.dimension))

    # <psi|A|psi> for a ket, tr(rho A) for a density operator rho, which also describes
    # mixtures. The two agree for rho = |psi><psi|.
    def expectation(self, state: StateSpaceVector | Operator) -> float:
        if not self.is_hermitian:
            raise ValueError("Expectation values are defined for Hermitian operators")
        if isinstance(state, Operator):
            if not state.is_density:
                raise ValueError("Expectation values require a normalized ket or a density")
            return (state @ self).trace().real
        if not (state.is_ket and state.is_normalized):
            raise ValueError("Expectation values require a normalized ket or a density")
        return (state.dagger @ (self @ state)).real

    def close_to(self, other: Operator, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        if self.state_space != other.state_space:
            return False
        return nearly_equal(self.matrix, other.matrix, tolerance)

    def __repr__(self) -> str:
        return f"Operator on {self.state_space}\n{self.matrix}"
