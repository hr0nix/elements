import functools
from collections.abc import Callable


from elements.quantum.state import StateSpace, StateSpaceVector
from elements.quantum.operator import Operator


def ensure_unitary[**P](f: Callable[P, Operator]) -> Callable[P, Operator]:
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Operator:
        result = f(*args, **kwargs)
        if not result.is_unitary:
            raise ValueError(f"{f.__qualname__} must return a unitary operator")
        return result
    return wrapper


class Gates:
    @staticmethod
    @ensure_unitary
    def flip(space: StateSpace) -> Operator:
        assert len(space.labels) == 2
        plus = space.ket(space.labels[0])
        minus = space.ket(space.labels[1])
        return plus @ minus.dagger + minus @ plus.dagger

    @staticmethod
    @ensure_unitary
    def controlled(apply_on: StateSpaceVector, target: Operator):
        on_projector = Operator.density(apply_on)
        off_projector = Operator.identity(apply_on.state_space) - on_projector
        return (
            on_projector.tensor_product(target) +
            off_projector.tensor_product(Operator.identity(target.state_space))
        )

