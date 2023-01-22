from Layers import InLayer, Layer
from typing import Self
import numpy as np
from utils import mul, diag
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
        self.derivs = [np.empty((self.layers[i+1], self.layers[i])).astype(float) for i in range(self.depth-2)]
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
        for m in self.derivs:
            m[:] = np.nan
    def cost(self) -> float|None:
        if self.answer is None:
            return None
        self.layers[-1].cost(self.answer)
    def set_input(self, ninput: np.matrix, normalize: bool = False) -> Self:
        """ninput: (1, width)"""
        self.layers[0].set_answers(ninput, normalize)
        for m in self.derivs:
            m[:] = np.nan
        return self
    def __call__(self) -> np.matrix:
        if any(self.derivs, key=lambda m:np.isnan(m).any()):
            for i in range(len(self.derivs)):
                self.derivs[i] = self.deriv_between(i+2)
        return self.layers[-1]()
    def deriv_between(self, right: int) -> np.matrix:
        """
        return: (right.width, right.prev.width)#TODO
        derivetive between two adjesent layers
        """
        right = self.layers[right]
        d1 = mul(right(), 1-right()) #(1, right)
        d2 = self.layers[right].weights #(prev, right)
        return np.dot(d2, diag(d1))
    def gradient(self) -> tuple[list[np.matrix], list[np.matrix]]:
        """
        layer_i: index of layer which bias is deriviated
        return: list(1, width_of_i)
        """
        bias: list[np.matrix] = []
        weight: list[np.matrix] = []
        overall = np.identity(self.layers[-1].width) #(layers[-1].width, layers[layer_i].width)
        for i in range(self.depth-1, 1, -1):
            layer = self.layers[i]
            tmp_deriv = mul(layer(), 1-layer())
            deriv = self.derivs[i-2]
            bias.append(mul(overall.sum(0), tmp_deriv))
            weight.append(np.dot(self.layers[i-1]().T, tmp_deriv))
            overall = np.dot(overall, deriv)
        layer = self.layers[1]
        tmp_deriv = mul(layer(), 1-layer())
        bias.append(mul(overall.sum(0), tmp_deriv))
        weight.append(np.dot(self.layers[0]().T, tmp_deriv))
        return bias, weight #list[d-1](1, layers[layer_i].width), list[d-1]()
    def descent(self) -> Self:
        if any(self.derivs, key=lambda m:np.isnan(m).any()):
            self()
        dJ_db, dJ_dw = self.gradient()
        for layer, dJ_dbi, dJ_dwi in zip(self.layers[1:], dJ_db, dJ_dw):
            layer.descent(dJ_dbi, dJ_dwi)
        for m in self.derivs:
            m[:] = np.nan
        return self