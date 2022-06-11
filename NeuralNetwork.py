from itertools import zip_longest
from typing import Tuple, Any, List

import numpy as np
from numpy import ndarray

from FullyConnectedLayer import FullyConnectedLayer
from Sigmoid import Sigmoid


class NeuralNetwork:

    def __init__(self, layers: list[FullyConnectedLayer], activations: list[Sigmoid]):
        self.layers = layers
        self.activations = activations
        # We save the layer and the activation inputs of each layer to use later in the backward fct
        self.layer_inputs = []
        self.activation_inputs = []

    def forward(self, x: np.array)\
            -> np.array:
        for layer, activation in zip_longest(self.layers, self.activations):
            self.layer_inputs.append(x)
            x = layer.forward(x)
            if activation is not None:
                self.activation_inputs.append(x)
                x = activation.forward(x)
        return x

    def backward(self, grad: np.array = np.array([[1]]))\
            -> tuple[ndarray, list[ndarray], list[ndarray]]:
        # input_gradient, weight_gradients, bias_gradients
        w_grads, b_grads = [], []
        # last layer does not have an activation function so we need to append a None element
        for layer, activation, layer_input, activation_input in \
                zip_longest(reversed(self.layers), reversed(self.activations + [None]),
                            reversed(self.layer_inputs), reversed(self.activation_inputs + [None])):
            if activation is not None:
                grad = activation.backward(activation_input, grad)
            grad, w_grad, b_grad = layer.backward(layer_input, grad)
            w_grads.append(w_grad)
            b_grads.append(b_grad)

        return grad, list(reversed(w_grads)), list(reversed(b_grads))
