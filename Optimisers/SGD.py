from numpy import ndarray

from Layers.FullyConnected import FullyConnected
from NeuralNetwork import NeuralNetwork


class SGD:

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float):
        self.learning_rate = learning_rate
        # Filter all Layers with weights and biases, in our case fcn layers at the moment
        self.fcn_layers = []
        for layer in neural_network.layers:
            if isinstance(layer, FullyConnected):
                self.fcn_layers.append(layer)  # hopefully this is a reference!

    def update(self, w_grads: list[ndarray], b_grads: list[ndarray]) -> None:
        # Do the Vanilla Stochastic Gradient Descent
        for layer, w_grad, b_grad in zip(self.fcn_layers, w_grads, b_grads):
            layer.weights += - self.learning_rate * w_grad
            layer.bias += - self.learning_rate * b_grad
