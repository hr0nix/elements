from elements.quantum.state import StateSpaceVector
from elements.quantum.operator import Operator


def density(state: StateSpaceVector) -> Operator:
    return state @ state.dagger


def purity(dens: Operator) -> float:
    return (dens @ dens).trace().real


def expectation(dens: Operator, operator: Operator) -> float:
    return (dens @ operator).trace().real
