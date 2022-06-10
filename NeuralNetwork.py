from itertools import zip_longest

import numpy as np

from FullyConnectedLayer import FullyConnectedLayer
from Sigmoid import Sigmoid


class NeuralNetwork:

    def __init__(self, layers: list[FullyConnectedLayer], activations: list[Sigmoid]):
        self.layers = layers
        self.activations = activations

    def forward(self, x: np.array)\
            -> np.array:
        for layer, activation in zip_longest(self.layers, self.activations):
            x = layer.forward(x)
            if activation is not None:
                x = activation.forward(x)
        return x

    def backward(self, x: np.array, grad: np.array = np.array([[1]]))\
            -> tuple[np.array]:
        pass
