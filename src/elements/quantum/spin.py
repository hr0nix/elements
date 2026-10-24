import numpy as np

from elements.quantum.operator import Operator
from elements.quantum.state import StateSpace


class Spin:
    state_space = StateSpace("u", "d")

    z_plus = state_space.ket("u")
    z_minus = state_space.ket("d")
    x_plus = (z_plus + z_minus) / np.sqrt(2)
    x_minus = (z_plus - z_minus) / np.sqrt(2)
    y_plus = (z_plus + 1j * z_minus) / np.sqrt(2)
    y_minus = (z_plus - 1j * z_minus) / np.sqrt(2)
    up = z_plus
    down = z_minus

    identity = Operator.identity(state_space)
    sigma_x = Operator.spectral([(+1, x_plus), (-1, x_minus)])
    sigma_y = Operator.spectral([(+1, y_plus), (-1, y_minus)])
    sigma_z = Operator.spectral([(+1, z_plus), (-1, z_minus)])


class TwoSpins:
    state_space = Spin.state_space.tensor_product(Spin.state_space)

    uu = Spin.up.tensor_product(Spin.up)
    ud = Spin.up.tensor_product(Spin.down)
    du = Spin.down.tensor_product(Spin.up)
    dd = Spin.down.tensor_product(Spin.down)

    singlet = (ud - du) / np.sqrt(2)
    triplet_1 = (ud + du) / np.sqrt(2)
    triplet_2 = (uu + dd) / np.sqrt(2)
    triplet_3 = (uu - dd) / np.sqrt(2)
