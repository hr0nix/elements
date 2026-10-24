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
        self._eigenbasis = hamiltonian.eigenbasis
        self._energies = hamiltonian.eigenvalues
        self._components = frozen(self._eigenbasis.coordinates(initial))

    # Energies of the eigenstates, in increasing order.
    @property
    def energies(self) -> np.ndarray:
        return self._energies

    # Components of the initial state along the eigenstates, in the order of energies.
    @property
    def energy_components(self) -> np.ndarray:
        return self._components

    # Each energy eigenstate |E> only turns its phase, to exp(-i E t) |E>.
    def state_at(self, time: float) -> StateSpaceVector:
        return self._eigenbasis.ket(np.exp(-1j * self._energies * time) * self._components)
