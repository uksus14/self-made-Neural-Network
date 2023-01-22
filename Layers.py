from typing import Self
import numpy as np
from utils import sigmoid, mul, uni_default
__network_debug__ = False
class InLayer:pass
__layers__ = []
class Layer:
    def __init__(self, prev: Self|InLayer, nodes: np.matrix, weights: np.matrix) -> None:
        """
        nodes: (2, width) of answers in first column and bias in second
        weights: (prev_width, width) of weights after the layer
        """
        __layers__.append(self)
        self.depth = prev.depth+1
        self.prev_width, self.width = weights.shape
        self.prev = prev
        self.nodes = nodes.astype(float)
        self.weights = weights.astype(float)
    @staticmethod
    def new(prev: Self|InLayer, width: int, rand_weights: slice|None = None, rand_biases: slice|None = None) -> Self:
        nodes = np.empty((2, width))
        rand = uni_default(rand_biases, 0)
        nodes[1, :] = rand(nodes[1, :])
        rand = uni_default(rand_weights, 1)
        weights = rand(np.empty((prev.width, width)))
        return Layer(prev, nodes, weights)
    def clear_answers(self) -> Self:
        self.nodes[0, :] = np.empty((1, self.width))
        next = [layer for layer in __layers__ if layer.prev==self]
        if next: next[0].clear_answers()
        return self
    def clear(self) -> Self:
        self.nodes[1, :] = np.zeros((1, self.width))
        self.weights = np.ones((self.prev_width, self.width))
        return self.clear_answers()
    def __call__(self) -> np.matrix: # (1, width)
        if np.any(np.isnan(self.nodes[0, :])):
            self.generate()
        return self.nodes[0, :]
    def generate(self) -> Self:
        answers = np.dot(self.prev(), self.weights) #(1, width)
        activation = sigmoid(answers+self.nodes[1, :]) #(1, width)
        self.nodes[0, :] = activation
        return self
    def set_answers(self, answers: np.matrix, normalize: bool = False) -> Self:
        """answers: (1, width)"""
        if normalize: answers = sigmoid(answers)
        self.nodes[0, :] = answers
        next = [layer for layer in __layers__ if layer.prev==self]
        if next: next[0].clear_answers()
        return self
    def create_answer(self, answer: int) -> np.matrix:
        """
        answer: index of right answer neuron
        return: (1, width) like [0,0,...1...0,0]
        """
        out = np.zeros((1, self.width))
        out[0, answer] = 1
        return out
    def cost(self, answer: int|np.matrix) -> float:
        """
        answer: index of right answer neuron or return of create_answer
        return: cost of network
        """
        if isinstance(answer, int):
            answer = self.create_answer(answer)
        loss = self()-answer
        return mul(loss, loss).sum()
    def descent(self, dJ_db: np.matrix, dJ_dw: np.matrix) -> Self:
        self.nodes[1, :] -= dJ_db
        self.weights -= dJ_dw
        return self
class InLayer(Layer):
    def __init__(self, nodes: np.matrix) -> None:
        """
        nodes: (1, width) of answers
        """
        __layers__.append(self)
        self.depth = 0
        _,self.width = nodes.shape
        self.nodes = nodes.astype(float)
        self.prev = None
    @staticmethod
    def new(width: int) -> Self:
        return InLayer(np.zeros((1, width)))
    def clear(self) -> Self: return self.clear_answers()
    def generate(self) -> Self: return self