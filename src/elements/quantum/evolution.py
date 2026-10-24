from elements.quantum.operator import Operator
from elements.quantum.state import StateSpaceVector


def evolve(operator: Operator, state: StateSpaceVector, time: float) -> StateSpaceVector:
    return operator.exp(-1j * time) @ state