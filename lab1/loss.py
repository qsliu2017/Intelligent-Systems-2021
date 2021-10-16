from numpy import ndarray
import numpy as np

from util import assert_same_shape


class Loss:
    def forward(self, prediction: ndarray, target: ndarray) -> float:
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        return (
            np.sum(np.power(self.prediction - self.target, 2)) /
            self.prediction.shape[0]
        )

    def backward(self) -> ndarray:
        self.input_grad = (
            2.0 * (self.prediction - self.target) /
            self.prediction.shape[0]
        )
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad
