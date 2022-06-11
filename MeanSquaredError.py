import numpy as np


class MeanSquaredError:

    def __init__(self):
        pass

    def forward(self, y_pred: np.array, y_true: np.array)\
            -> float:
        return 1/len(y_pred) * np.sum(0.5 * (y_true - y_pred) ** 2)

    def backward(self, y_pred: np.array, y_true: np.array, grad: np.array = np.array([[1]]))\
            -> np.array:
        return grad * (y_pred - y_true)
