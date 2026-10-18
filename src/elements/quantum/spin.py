import numpy as np

from elements.quantum.state import StateSpace, StateSpaceVector


class Spin:
    @staticmethod
    def state_space() -> StateSpace:
        return StateSpace(labels=["u", "d"])

    @staticmethod
    def z_plus() -> StateSpaceVector:
        return StateSpaceVector(
            state_space=Spin.state_space(), components=np.array([[1.0], [0.0]], dtype=complex)
        )

    @staticmethod
    def z_minus() -> StateSpaceVector:
        return StateSpaceVector(
            state_space=Spin.state_space(), components=np.array([[0.0], [1.0]], dtype=complex)
        )

    @staticmethod
    def x_plus() -> StateSpaceVector:
        return (Spin.z_plus() + Spin.z_minus()) / np.sqrt(2)

    @staticmethod
    def x_minus() -> StateSpaceVector:
        return (Spin.z_plus() - Spin.z_minus()) / np.sqrt(2)

    @staticmethod
    def y_plus() -> StateSpaceVector:
        return (Spin.z_plus() + 1j * Spin.z_minus()) / np.sqrt(2)

    @staticmethod
    def y_minus() -> StateSpaceVector:
        return (Spin.z_plus() - 1j * Spin.z_minus()) / np.sqrt(2)


class TwoSpins:
    @staticmethod
    def state_space() -> StateSpace:
        return Spin.state_space().tensor_product(Spin.state_space())

    @staticmethod
    def uu() -> StateSpaceVector:
        return Spin.z_plus().kronecker_product(Spin.z_plus())

    @staticmethod
    def ud() -> StateSpaceVector:
        return Spin.z_plus().kronecker_product(Spin.z_minus())

    @staticmethod
    def du() -> StateSpaceVector:
        return Spin.z_minus().kronecker_product(Spin.z_plus())

    @staticmethod
    def dd() -> StateSpaceVector:
        return Spin.z_minus().kronecker_product(Spin.z_minus())

    @staticmethod
    def singlet() -> StateSpaceVector:
        return (TwoSpins.ud() - TwoSpins.du()) / np.sqrt(2)

    @staticmethod
    def triplet_1() -> StateSpaceVector:
        return (TwoSpins.ud() + TwoSpins.du()) / np.sqrt(2)

    @staticmethod
    def triplet_2() -> StateSpaceVector:
        return (TwoSpins.uu() + TwoSpins.dd()) / np.sqrt(2)

    @staticmethod
    def triplet_3() -> StateSpaceVector:
        return (TwoSpins.uu() - TwoSpins.dd()) / np.sqrt(2)
