import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
def mul(*args: np.matrix) -> np.matrix:
    """
    get matricis with same shapes and multiplies element wise
    """
    return np.multiply(mul(*args[:-1]), args[-1]) if len(args)-1 else args[0]
def rand_in_slice(s: slice| None, default: float|None) -> float:
    rand = lambda:0
    if s is not None:
        rand = lambda:np.random.uniform(s.start, s.stop)
    return rand
if __name__ == "__main__":
    rand = rand_in_slice(slice(-10, 10), None)
    while 1:print(eval(input()))