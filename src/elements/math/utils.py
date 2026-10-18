import numpy as np


def frozen(array: np.ndarray) -> np.ndarray:
    array = array.copy()
    array.flags.writeable = False
    return array
