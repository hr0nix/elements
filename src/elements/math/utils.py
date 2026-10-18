import numpy as np

DEFAULT_TOLERANCE = 1e-8


def frozen(array: np.ndarray) -> np.ndarray:
    array = array.copy()
    array.flags.writeable = False
    return array
