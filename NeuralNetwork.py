import numpy as np
from numpy import ndarray

from Layers.Layer import Layer


class NeuralNetwork:

    def __init__(self, layers: list[Layer]):
        self.layers = layers
        # We save the inputs of each layer to use later in the backward fct
        self.layer_inputs = []

    def forward(self, x: np.array)\
            -> np.array:
        for layer in self.layers:
            self.layer_inputs.append(x)
            x = layer.forward(x)
        return x

    def backward(self, grad: np.array = np.array([[1]]))\
            -> tuple[ndarray, list[ndarray], list[ndarray]]:
        # input_gradient, weight_gradients, bias_gradients
        w_grads, b_grads = [], []
        for layer, layer_input in zip(reversed(self.layers), reversed(self.layer_inputs)):
            grad, w_grad, b_grad = layer.backward(layer_input, grad)
            if w_grad is not None:
                w_grads.append(w_grad)
            if b_grad is not None:
                b_grads.append(b_grad)

        return grad, list(reversed(w_grads)), list(reversed(b_grads))
