import numpy as np

DEFAULT_TOLERANCE = 1e-8


def frozen(array: np.ndarray) -> np.ndarray:
    array = array.copy()
    array.flags.writeable = False
    return array


# Entries may differ by tolerance times the largest entry on either side, or by tolerance itself
# when every entry is below 1. Rounding error grows with the entries, so a fixed bound would
# fail large operators, such as Hamiltonians on fine grids, that are equal in exact arithmetic.
def nearly_equal(a: np.ndarray, b: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    scale = max(1.0, np.abs(a).max(initial=0.0), np.abs(b).max(initial=0.0))
    return bool(np.all(np.abs(a - b) <= tolerance * scale))
