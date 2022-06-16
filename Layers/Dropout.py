import numpy as np
from numpy import ndarray

from Layers.Layer import Layer


class Dropout(Layer):

    def __init__(self, p: float = 0.5):
        self.mask = None
        self.p = p

    def forward(self, x: np.array)\
            -> np.array:
        # We scale up while training! (See Chapter 3 p.70)
        self.mask = np.random.binomial(1, self.p, size=x.shape) / self.p
        print(self.mask)
        return x * self.mask

    def backward(self, x: np.array, grad: np.array) \
            -> tuple[ndarray, None, None]:
        return grad * self.mask, None, None
