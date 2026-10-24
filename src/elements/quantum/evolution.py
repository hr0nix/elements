import numpy as np

from elements.math.utils import frozen
from elements.quantum.operator import Operator
from elements.quantum.state import StateSpaceVector


class EigenbasisEvolution:
    # Expands the initial state in energy eigenstates once, so that every later state costs a
    # matrix-vector product instead of a fresh exponential of the Hamiltonian.
    def __init__(self, hamiltonian: Operator, initial: StateSpaceVector):
        if not initial.is_ket or initial.state_space != hamiltonian.state_space:
            raise ValueError("Initial state must be a ket from the Hamiltonian's state space")
        if not hamiltonian.is_hermitian:
            raise ValueError("Evolution needs a Hermitian Hamiltonian")
        self.state_space = hamiltonian.state_space
        self._energies = frozen(hamiltonian.eigenvalues.real)
        self._eigenbasis = np.hstack([vector.components for vector in hamiltonian.eigenvectors])
        self._components = frozen((self._eigenbasis.conj().T @ initial.components).ravel())

    # Energies of the eigenstates, in no particular order.
    @property
    def energies(self) -> np.ndarray:
        return self._energies

    # Components of the initial state along the eigenstates, in the order of energies.
    @property
    def energy_components(self) -> np.ndarray:
        return self._components

    def state_at(self, time: float) -> StateSpaceVector:
        components = np.exp(-1j * self._energies * time) * self._components
        return StateSpaceVector(self.state_space, self._eigenbasis @ components[:, np.newaxis])
