from utils import mul
import numpy as np
from Layers import Layer, InLayer
# first: InLayer = InLayer.new(3)
# second: Layer = Layer.new(first, 2)
# first.set_answers(np.ones((1, 3)))
# answer = second.create_answer(1)
# cost = second.cost(answer)
# print(second())
# print(cost)
# dJ_db = 2*mul(second()-answer, second(), 1-second())
# second.descent(dJ_db, np.zeros((first.width, second.width)))
# second.clear_answers()
# print(second())
# cost = second.cost(answer)
# print(cost)

print(np.ones((2, 3)).sum(0))