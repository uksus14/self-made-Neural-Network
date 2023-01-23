import numpy as np
from types import FunctionType
def sigmoid(z):
    return 1/(1+np.exp(-z))
def mul(*args: np.matrix) -> np.matrix:
    """
    get matricis with same shapes and multiplies element wise
    """
    return np.multiply(mul(*args[:-1]), args[-1]) if len(args)-1 else args[0]
def uni_default(s: slice| None, default: float|None) -> FunctionType:
    rand = lambda _:default
    if s is not None:
        rand = lambda _:np.random.uniform(s.start, s.stop)
    return np.vectorize(rand)
def diag(m: np.matrix) -> np.matrix:
    """
    m: (1, x)
    return: (x, x) with it's elements diagonalized
    """
    x = max(m.shape)
    answer = np.empty((x, x))
    np.fill_diagonal(answer, m)
    return answer