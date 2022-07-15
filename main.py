import numpy as np
from tqdm import tqdm

from Layers.Dropout import Dropout
from Layers.FullyConnected import FullyConnected
from MeanSquaredError import MeanSquaredError
from NeuralNetwork import NeuralNetwork
from Layers.Sigmoid import Sigmoid
from Optimisers.Adam import Adam
from Optimisers.Momentum import Momentum
from Optimisers.SGD import SGD


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

    # Optimiser
    # optimiser = SGD(net, learning_rate=0.1)
    # optimiser = Momentum(net, learning_rate=0.1, mu=0.01)
    optimiser = Adam(net, learning_rate=0.01, beta_1=0.9, beta_2=0.99)

    # Input
    # XOR Dataset
    inputs = [
        (np.array([[0, 0]]), np.array([0])),
        (np.array([[0, 1]]), np.array([1])),
        (np.array([[1, 0]]), np.array([1])),
        (np.array([[0, 0]]), np.array([0]))
    ]

    epochs = 50000
    for epoch in tqdm(range(epochs)):
        # It is always a good idea to shuffle the dataset
        np.random.shuffle(inputs)

        for x, y in inputs:
            # Forward Pass
            pred = net.forward(x)

            # Loss Calculation
            loss = loss_fct.forward(pred, y)

            # Backward Pass
            grad = loss_fct.backward(pred, y)
            grad, w_grads, b_grads = net.backward(grad)

            optimiser.update(w_grads, b_grads)

    # Test that the network has learned something
    for x, y in inputs:
        # Forward Pass
        pred = net.forward(x)

        print(f"Input: {x}, Output: {pred}, Desired Output: {y}")

    # # Input
    # x = np.array([[1, 1]])
    # y = np.array([[0]])
    #
    # # we simply apply our forward and backward calculations multiple times to see how they change!
    #
    # for i in range(10):
    #     # Forward Pass
    #     pred = net.forward(x)
    #
    #     # Loss Calc
    #     loss = loss_fct.forward(y_pred=pred, y_true=y)
    #
    #     print(f"Prediction: {pred}")
    #     print(f"Loss: {loss}")
    #
    #     # Backward Pass
    #     grad = loss_fct.backward(pred, y)
    #     grad, w_grads, b_grads = net.backward(grad)
    #
    #     print(f"Gradients of the first fcn layer: W1: {w_grads[0]}, b1: {b_grads[0]}")
    #     print(f"Gradients of the second fcn layer: W2: {w_grads[1]}, b2 {b_grads[1]}")


if __name__ == '__main__':
    main()
