from numpy import ndarray

from Layers.FullyConnected import FullyConnected
from NeuralNetwork import NeuralNetwork


class Momentum:

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float, mu: float):
        self.learning_rate = learning_rate
        self.mu = mu

        # We filter the necessary layers from our network
        self.fcn_layers = []
        for layer in neural_network.layers:
            if isinstance(layer, FullyConnected):
                self.fcn_layers.append(layer)  # hopefully this is a reference!

        self.v_ws = [0] * len(self.fcn_layers)
        self.v_bs = [0] * len(self.fcn_layers)

    def update(self, w_grads: list[ndarray], b_grads: list[ndarray])\
            -> None:
        # Do the Momentum Gradient Descent
        for layer, w_grad, b_grad, v_w, v_b in \
                zip(self.fcn_layers, w_grads, b_grads, self.v_ws, self.v_bs):
            v_w = self.mu * v_w - self.learning_rate * w_grad
            layer.weights += v_w
            v_b = self.mu * v_b - self.learning_rate * b_grad
            layer.bias += v_b

