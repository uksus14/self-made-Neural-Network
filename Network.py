from Layers import InLayer, Layer
from typing import Self
import numpy as np
from utils import mul, diag
class Network:
    def __init__(self, layers: list[Layer|InLayer]|Layer):
        """
        layers: list of all layers of the last one
        """
        if isinstance(layers, Layer):
            tmp = []
            layer = layers
            while layer:
                tmp.append(layer)
                layer = layer.prev
            layers = tmp[::-1]
        self.layers = layers
        self.depth = len(layers)
        self.width = [layer.width for layer in layers]
        self.answer = None
        self.derivs = [np.empty((self[i+1].width, self[i].width)).astype(float) for i in range(self.depth-2)]
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
            self.answer = self[-1].create_answer(answer)
        for m in self.derivs:
            m[:] = np.nan
    def cost(self) -> float|None:
        if self.answer is None:
            return None
        return self[-1].cost(self.answer)
    def set_input(self, ninput: np.matrix, normalize: bool = False) -> Self:
        """ninput: (width,)"""
        self[0].set_answers(ninput, normalize)
        for m in self.derivs:
            m[:] = np.nan
        return self
    def __call__(self) -> np.matrix:
        has_nan = lambda m:np.isnan(m).any()
        if any(map(has_nan, self.derivs)):
            for i in range(len(self.derivs)):
                self.derivs[i] = self.deriv_between(i+2)
        return self[-1]()
    def deriv_between(self, right: int) -> np.matrix:
        """
        return: (right.width, right.prev.width)#TODO
        derivetive between two adjesent layers
        """
        right: Layer = self[right]

        d1 = mul(right(), 1-right()) #(1, right)
        d2 = right.weights #(prev, right)
        return np.dot(d2, diag(d1))
    def gradient(self) -> tuple[list[np.matrix], list[np.matrix]]:
        """
        return: list[d-1](width_of_i,), list[d-1](width_of_i-1, width_of_i)
        where d is depth
        """
        bias: list[np.matrix] = []
        weight: list[np.matrix] = []
        last = self[-1]
        dJ_dlast = 2*(last()-self.answer)
        overall = np.identity(self[-1].width) #(layers[-1].width, layers[layer_i].width)
        for i in range(self.depth-1, 0, -1):
            layer = self[i]
            tmp_deriv = mul(layer(), 1-layer())
            top_deriv = mul(dJ_dlast, overall.sum(0))
            bias.append(mul(tmp_deriv, top_deriv).reshape((-1,)))
            weight.append(np.dot(np.dot(self[i-1]().T, tmp_deriv), diag(top_deriv)))
            try:
                deriv = self.derivs[i-2]
                overall = np.dot(overall, deriv)
            except IndexError:
                pass
        return bias[::-1], weight[::-1] #
    def descent(self) -> Self:
        has_nan = lambda m:np.isnan(m).any()
        if any(map(has_nan, self.derivs)):
            self()
        dJ_db, dJ_dw = self.gradient()
        layer: Layer
        for layer, dJ_dbi, dJ_dwi in zip(self[1:], dJ_db, dJ_dw):
            layer.descent(dJ_dbi, 0)
        for m in self.derivs:
            m[:] = np.nan
        return self
    def __getitem__(self, index):
        return self.layers[index]