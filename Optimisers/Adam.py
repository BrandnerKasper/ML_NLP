import numpy as np
from numpy import ndarray

from Layers.FullyConnected import FullyConnected
from NeuralNetwork import NeuralNetwork


class Adam:

    def __init__(self, neural_network: NeuralNetwork, learning_rate: float, beta_1: float, beta_2: float):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # We filter the necessary layers from our network
        self.fcn_layers = []
        for layer in neural_network.layers:
            if isinstance(layer, FullyConnected):
                self.fcn_layers.append(layer)  # hopefully this is a reference!

        self.m_ws = [0] * len(self.fcn_layers)
        self.m_bs = [0] * len(self.fcn_layers)
        self.v_ws = [0] * len(self.fcn_layers)
        self.v_bs = [0] * len(self.fcn_layers)

    def update(self, w_grads: list[ndarray], b_grads: list[ndarray]):
        for layer, m_w, v_w, w_grad, m_b, v_b, b_grad in \
                zip(self.fcn_layers, self.m_ws, self.v_ws, w_grads,
                    self.m_bs, self.v_bs, b_grads):
            m_w = self.beta_1 * m_w + (1 - self.beta_1) * w_grad
            v_w = self.beta_2 * v_w + (1 - self.beta_2) * w_grad**2
            layer.weights += - self.learning_rate / (np.sqrt(v_w) + 1e-8) * m_w

            m_b = self.beta_1 * m_b + (1 - self.beta_1) * b_grad
            v_b = self.beta_2 * v_b + (1 - self.beta_2) * b_grad ** 2
            layer.bias += - self.learning_rate / (np.sqrt(v_b) + 1e-8) * m_b
