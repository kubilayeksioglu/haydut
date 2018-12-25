import numpy as np


def argmax_tiebreak(arr):
    return np.random.choice(np.flatnonzero(arr == arr.max()))


def argmin_tiebreak(arr):
    return argmax_tiebreak(-arr)

