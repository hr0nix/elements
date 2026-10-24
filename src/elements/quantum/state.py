import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from elements.math.typechecks import is_bra_vector, is_ket_vector, is_vector
from elements.math.utils import DEFAULT_TOLERANCE, frozen
from elements.util.formatting import split_sign

if TYPE_CHECKING:
    from elements.quantum.operator import Operator


class StateSpace:
    # Factors are kept flat, so (A ⊗ B) ⊗ C and A ⊗ (B ⊗ C) are the same space.
    def __init__(self, labels: list[str], *, factors: Sequence[StateSpace] | None = None):
        if len(labels) == 0:
            raise ValueError("State space must have at least one label")
        if len(set(labels)) != len(labels):
            raise ValueError("Labels must be unique")
        self._labels = tuple(labels)

        if factors is None:
            self._factors = (self,)
            return
        if len(factors) == 0:
            raise ValueError("A product space needs at least one factor")
        self._factors = tuple(atom for factor in factors for atom in factor.factors)
        if math.prod(atom.dimension for atom in self._factors) != len(labels):
            raise ValueError("Number of labels must match the product of the factor dimensions")

    def tensor_product(self, other: StateSpace) -> StateSpace:
        factors = self.factors + other.factors
        combinations = list(itertools.product(*(factor.labels for factor in factors)))
        labels = ["".join(combination) for combination in combinations]
        # Plain concatenation can be ambiguous, e.g. "1" + "12" == "11" + "2".
        if len(set(labels)) != len(labels):
            labels = [",".join(combination) for combination in combinations]
        return StateSpace(labels, factors=factors)

    def __len__(self) -> int:
        return len(self._labels)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateSpace):
            return NotImplemented
        # Comparing the factors themselves would recurse forever: an elementary space is its own
        # factor. Factors are always elementary, so their labels identify them.
        factor_labels = [factor.labels for factor in self._factors]
        other_factor_labels = [factor.labels for factor in other._factors]
        return self._labels == other._labels and factor_labels == other_factor_labels

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def factors(self) -> tuple[StateSpace, ...]:
        return self._factors

    @property
    def dimension(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        return f"StateSpace({', '.join(f'|{label}>' for label in self._labels)})"


class StateSpaceVector:
    def __init__(self, state_space: StateSpace, components: np.ndarray):
        components = np.asarray(components, dtype=complex)
        if not is_vector(components):
            raise ValueError("Components must be a bra or ket vector")
        if components.size != state_space.dimension:
            raise ValueError("Number of coefficients must match number of labels")

        self.state_space = state_space
        self.components = frozen(components)

    @property
    def is_bra(self) -> bool:
        return is_bra_vector(self.components)

    @property
    def is_ket(self) -> bool:
        return is_ket_vector(self.components)

    def is_state_space_compatible(self, other: StateSpaceVector) -> bool:
        return self.state_space == other.state_space

    def is_bra_ket_compatible(self, other: StateSpaceVector) -> bool:
        return self.is_ket == other.is_ket

    def is_compatible(self, other: StateSpaceVector) -> bool:
        return self.is_state_space_compatible(other) and self.is_bra_ket_compatible(other)

    def close_to(self, other: StateSpaceVector, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        if not self.is_compatible(other):
            return False
        return np.allclose(self.components, other.components, rtol=0.0, atol=tolerance)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.components))

    @property
    def normalized(self) -> StateSpaceVector:
        return self * (1.0 / self.norm)

    @property
    def is_normalized(self) -> bool:
        return np.isclose(self.norm, 1.0)

    def __mul__(self, other: complex) -> StateSpaceVector:
        return StateSpaceVector(self.state_space, self.components * other)

    def __rmul__(self, other: complex) -> StateSpaceVector:
        return StateSpaceVector(self.state_space, self.components * other)

    def __truediv__(self, other: complex) -> StateSpaceVector:
        return self * (1 / other)

    def __neg__(self) -> StateSpaceVector:
        return (-1) * self

    def __add__(self, other: StateSpaceVector) -> StateSpaceVector:
        if not self.is_compatible(other):
            raise ValueError("State space vectors must be compatible")
        return StateSpaceVector(self.state_space, self.components + other.components)

    def __sub__(self, other: StateSpaceVector) -> StateSpaceVector:
        return self + (-1) * other

    def kronecker_product(self, other: StateSpaceVector) -> StateSpaceVector:
        if not self.is_bra_ket_compatible(other):
            raise ValueError("Kronecker product of states requires two kets or two bras")

        return StateSpaceVector(
            self.state_space.tensor_product(other.state_space),
            np.kron(self.components, other.components),
        )

    @property
    def dagger(self) -> StateSpaceVector:
        return StateSpaceVector(self.state_space, self.components.conj().T)

    def inner_product(self, other: StateSpaceVector) -> complex:
        if not self.is_state_space_compatible(other):
            raise ValueError("Inner product requires vectors from the same state space")
        if not (self.is_bra and other.is_ket):
            raise ValueError("Inner product requires a bra and a ket; use .dagger to convert")

        return (self.components @ other.components).item()

    def outer_product(self, other: StateSpaceVector) -> Operator:
        from elements.quantum.operator import Operator

        if not self.is_state_space_compatible(other):
            raise ValueError("Outer product requires vectors from the same state space")
        if not (self.is_ket and other.is_bra):
            raise ValueError("Outer product requires a ket and a bra; use .dagger to convert")

        return Operator(self.state_space, self.components @ other.components)

    def __matmul__(self, other: StateSpaceVector) -> complex | Operator:
        if not isinstance(other, StateSpaceVector):
            return NotImplemented
        if self.is_bra and other.is_ket:
            return self.inner_product(other)
        if self.is_ket and other.is_bra:
            return self.outer_product(other)
        raise ValueError("@ requires a bra and a ket; use .dagger to convert")

    def __repr__(self) -> str:
        left, right = ("|", ">") if self.is_ket else ("<", "|")
        terms = [
            (*split_sign(coefficient), label)
            for coefficient, label in zip(
                self.components.ravel(), self.state_space.labels, strict=True
            )
            if abs(coefficient) > DEFAULT_TOLERANCE
        ]
        if not terms:
            return "0"
        text = "".join(
            f" {sign} {magnitude}{left}{label}{right}" for sign, magnitude, label in terms
        )
        return text[3:] if text.startswith(" + ") else f"-{text[3:]}"
