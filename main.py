import numpy as np

from FullyConnectedLayer import FullyConnectedLayer
from MeanSquaredError import MeanSquaredError
from NeuralNetwork import NeuralNetwork
from Sigmoid import Sigmoid


def main():
    # Sizes Init
    input_size = 2
    hidden_size = 2
    output_size = 1

    # Init Layers
    fc_1 = FullyConnectedLayer(input_size, hidden_size)
    # Weights v
    v = np.array([[0.5, 0.75],
                  [0.25, 0.25]])
    fc_1.weights = v
    sigmoid_1 = Sigmoid()
    fc_2 = FullyConnectedLayer(hidden_size, output_size)
    # Weights w
    w = np.array([[0.5], [0.5]])
    fc_2.weights = w

    # Network Init
    net = NeuralNetwork([fc_1, fc_2], [sigmoid_1])

    # Loss
    loss_fct = MeanSquaredError()

    # Input
    x = np.array([[1, 1]])
    y = np.array([[0]])

    # Forward Pass
    pred = net.forward(x)

    # Loss Calc
    loss = loss_fct.forward(y_pred=pred, y_true=y)

    print(f"Prediction: {pred}")
    print(f"Loss: {loss}")

    # Backward Pass
    grad = loss_fct.backward(pred, y)
    grad, w_grads, b_grads = net.backward(grad)

    print(f"Gradients of the first layer: W1: {w_grads[0]}, b1: {b_grads[0]}")
    print(f"Gradients of the second layer: W2: {w_grads[1]}, b2 {b_grads[1]}")


if __name__ == '__main__':
    main()
