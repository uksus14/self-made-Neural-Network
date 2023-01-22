from Layers import InLayer, Layer
from typing import Self
import numpy as np
from utils import mul
class Network:
    def __init__(self, layers: list[Layer]|Layer):
        """
        layers: list of all layers of the last one
        """
        if isinstance(layers, Layer):
            tmp = []
            layer = layers
            while layer:
                tmp.append(layer)
                layer = layer.prev
            layers = tmp
        self.layers = layers
        self.depth = len(layers)
        self.width = [layer.width for layer in layers]
        self.answer = None
        self.derivs = [np.empty((1, self.layers[i])).astype(float) for i in range(1, self.depth-1)]
    @staticmethod
    def new(width: list[int], randomize: tuple[slice|None, slice|None] = (None, None)) -> Self:
        """
        width: (depth) width of every layer
        randomize: (slice|None, slice|None) first is random range of weights and second is biases
        """
        prev = InLayer.new(width[0])
        for w in width[1:]:
            prev = Layer.new(prev, w, *randomize)
        return Network(prev)
    def set_answer(self, answer: int|None) -> Self:
        self.answer = None
        if answer is not None:
            self.answer = self.layers[-1].create_answer(answer)
    def cost(self) -> float|None:
        if self.answer is None:
            return None
        self.layers[-1].cost(self.answer)
    def set_input(self, ninput: np.matrix, normalize: bool = False) -> Self:
        """ninput: (1, width)"""
        self.layers[0].set_answers(ninput, normalize)
        self.derivs[:] = np.nan
        return self
    def __call__(self) -> np.matrix:
        return self.layers[-1]()
    def deriv_between(self, left: int) -> np.matrix:
        """
        return: (1, left.width)
        derivetive between two adjesent layers
        """
        left, right = self.layers[left], self.layers[left+1]
        return np.dot(mul(right(), 1-right()), right.weights.T)
    def deriv_from(self, left: int) -> np.matrix:
        """
        return: (1, left.width)
        """
    def deriv_bias(self, layer_i: int) -> np.matrix:
        """
        layer_i: index of layer which bias is deriviated
        return: (1, width_of_i)
        """
        layer = self.layers[layer_i]
        return mul(layer(), 1-layer())
        self.derivs[:] = np.nan
        