import numpy as np


class Sigmoid:

    def __int__(self):
        pass

    def forward(self, x: np.array)\
            -> np.array:
        return 1 / (1 + np.exp(-x))

    def backward(self, x: np.array, grad: np.array = np.array([[1]]))\
            -> np.array:
        pass
