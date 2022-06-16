from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.array)\
            -> np.array:
        pass

    @abstractmethod
    def backward(self, x: np.array, grad: np.array)\
            -> tuple[ndarray, ndarray, ndarray]:
        pass
