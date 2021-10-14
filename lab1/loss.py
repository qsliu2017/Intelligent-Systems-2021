from numpy import ndarray
import numpy as np


class Loss:
    def forward(self, prediction: ndarray, target: ndarray) -> float:
        self.prediction = prediction
        self.target = target
        return np.sum(np.power(self.prediction - self.target, 2)) / 2.0

    def backward(self) -> ndarray:
        return self.prediction - self.target
