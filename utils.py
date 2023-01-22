import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
def mul(*args: np.matrix) -> np.matrix:
    """
    get matricis with same shapes and multiplies element wise
    """
    return np.multiply(mul(*args[:-1]), args[-1]) if len(args)-1 else args[0]
def uni_default(s: slice| None, default: float|None) -> function:
    rand = lambda _:default
    if s is not None:
        rand = lambda _:np.random.uniform(s.start, s.stop)
    return np.vectorize(rand)