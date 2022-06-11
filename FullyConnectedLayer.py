from typing import Tuple

import numpy as np
from numpy import ndarray


class FullyConnectedLayer:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(self.input_size, self.output_size)
        self.bias = np.zeros((1, self.output_size))

    def forward(self, x: np.array) \
            -> np.array:
        return np.matmul(x, self.weights) + self.bias

    def backward(self, x: np.array, grad: np.array = np.array([[1]])) \
            -> tuple[ndarray, ndarray, ndarray]:
        # grad_input, grad_weights, grad_bias
        return np.matmul(grad, self.weights.T), np.matmul(x.T, grad), grad