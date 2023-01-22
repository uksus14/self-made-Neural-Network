from Layers import InLayer, Layer
from typing import Self
import numpy as np
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
        return self
    def __call__(self) -> np.matrix:
        return self.layers[-1]()
    def dJ_db(self, layer_i: int) -> np.matrix:
        """
        layer_i: index of layer which bias is deriviated
        return: (1, width_of_i)
        """
        