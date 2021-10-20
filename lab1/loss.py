from numpy import ndarray
import numpy as np

from util import assert_same_shape
from scipy.special import softmax


class Loss:
    def forward(self, prediction: ndarray, target: ndarray) -> float:
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        return self._output()

    def backward(self) -> ndarray:
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)
        return self.input_grad

    def _output(self) -> float:
        raise NotImplementedError()

    def _input_grad(self) -> ndarray:
        return NotImplementedError()


class MeanSquareLoss(Loss):
    def _output(self) -> float:
        return (
            np.sum(np.power(self.prediction - self.target, 2)) /
            self.prediction.shape[0]
        )

    def _input_grad(self) -> ndarray:
        return (
            2.0 * (self.prediction - self.target) /
            self.prediction.shape[0]
        )


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def _output(self) -> float:
        self.prediction_softmax = softmax(self.prediction, axis=1)
        self.prediction_softmax = np.clip(
            self.prediction_softmax, self.eps, 1-self.eps)
        self.softmax_cross_entropy_loss = -1.0 * \
            self.target * np.log(self.prediction_softmax)
        return np.sum(self.softmax_cross_entropy_loss) / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        return (self.prediction_softmax - self.target) / self.prediction.shape[0]
