import numpy as np
from numpy import ndarray

from Layers.Layer import Layer


class Sigmoid(Layer):

    def __int__(self):
        pass

    def forward(self, x: np.array)\
            -> np.array:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array, grad: np.array = np.array([[1]]))\
            -> tuple[ndarray, None, None]:
        # grad_input, grad_weights = None, grad_bias = None -> Sigmoid does not have weight and bias
        return grad * self.forward(x) * (1 - self.forward(x)), None, None
