import numpy as np

from Layers.Dropout import Dropout
from Layers.FullyConnected import FullyConnected
from MeanSquaredError import MeanSquaredError
from NeuralNetwork import NeuralNetwork
from Layers.Sigmoid import Sigmoid


def main():
    # Sizes Init
    input_size = 2
    hidden_size = 2
    output_size = 1

    # Init Layers
    dropout_1 = Dropout(0.5)
    fc_1 = FullyConnected(input_size, hidden_size)
    # Weights v -> we set weights manually according to the example
    v = np.array([[0.5, 0.75],
                  [0.25, 0.25]])
    fc_1.weights = v
    sigmoid_1 = Sigmoid()
    dropout_2 = Dropout(0.5)
    fc_2 = FullyConnected(hidden_size, output_size)
    # Weights w
    w = np.array([[0.5], [0.5]])
    fc_2.weights = w

    # Network Init
    net = NeuralNetwork([dropout_1, fc_1, sigmoid_1, dropout_2, fc_2])

    # Loss
    loss_fct = MeanSquaredError()

    # Input
    x = np.array([[1, 1]])
    y = np.array([[0]])

    # we simply apply our forward and backward calculations multiple times to see how they change!

    for i in range(10):
        # Forward Pass
        pred = net.forward(x)

        # Loss Calc
        loss = loss_fct.forward(y_pred=pred, y_true=y)

        print(f"Prediction: {pred}")
        print(f"Loss: {loss}")

        # Backward Pass
        grad = loss_fct.backward(pred, y)
        grad, w_grads, b_grads = net.backward(grad)

        print(f"Gradients of the first fcn layer: W1: {w_grads[0]}, b1: {b_grads[0]}")
        print(f"Gradients of the second fcn layer: W2: {w_grads[1]}, b2 {b_grads[1]}")


if __name__ == '__main__':
    main()
