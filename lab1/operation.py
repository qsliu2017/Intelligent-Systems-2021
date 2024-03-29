import numpy as np
from numpy import ndarray

from util import assert_same_shape


class Operation:
    def forward(self, input: ndarray):
        self.input = input
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        assert_same_shape(self.input, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()


class Sigmod(Operation):
    def _output(self) -> ndarray:
        return 1.0 / (1.0+np.exp(-self.input))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return self.output * (1.0-self.output) * output_grad


class Linear(Operation):
    def _output(self) -> ndarray:
        return self.input

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad


class Tanh(Operation):
    def _output(self) -> ndarray:
        return np.tanh(self.input)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * (1 - self.output * self.output)


class ParamOperation(Operation):
    def __init__(self, param: ndarray):
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        self.param_grad = self._param_grad(output_grad)
        assert_same_shape(self.param, self.param_grad)
        return super().backward(output_grad)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self) -> ndarray:
        return np.dot(self.input, self.param)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, self.param.T)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(self.input.T, output_grad)


class BiasAdd(ParamOperation):
    def __init__(self, B: ndarray):
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self) -> ndarray:
        return self.input + self.param

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(np.ones((1, self.input.shape[0])), output_grad)
