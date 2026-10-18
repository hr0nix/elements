import numpy as np


def is_complex(array: np.ndarray) -> bool:
    return array.dtype == complex


def is_vector(array: np.ndarray) -> bool:
    return array.ndim == 2 and 1 in array.shape


def is_complex_vector(array: np.ndarray) -> bool:
    return is_complex(array) and is_vector(array)


def is_bra_vector(array: np.ndarray) -> bool:
    return is_complex_vector(array) and array.shape[0] == 1


def is_ket_vector(array: np.ndarray) -> bool:
    return is_complex_vector(array) and array.shape[1] == 1


def is_real(array: np.ndarray) -> bool:
    if array.dtype in [float, int, np.float32, np.float64, np.int32, np.int64]:
        return True

    if array.dtype in [complex, np.complex64, np.complex128]:
        return np.isclose(array.imag, 0.0)

    return False
