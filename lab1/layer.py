import numpy as np
from numpy import ndarray
from util import assert_same_shape
from operation import Operation, ParamOperation, BiasAdd, WeightMultiply


class Layer(object):
    def __init__(self):
        self.first = True
        self.params: list[ndarray] = []
        self.param_grads: list[ndarray] = []
        self.operations: list[Operation] = []

    def _setup_layer(self, input: ndarray):
        raise NotImplementedError()

    def forward(self, input: ndarray) -> ndarray:
        if self.first:
            self._setup_layer(input)
            self.first = False
        self.input = input
        for operation in self.operations:
            input = operation.forward(input)
        self.output = input
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        self._param_grads()
        return output_grad

    def _param_grads(self):
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self):
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)


class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation):
        super().__init__()
        self.neurons = neurons
        self.activation = activation

    def _setup_layer(self, input: ndarray):
        if self.seed:
            np.random.seed(self.seed)
        self.params = []
        # weight multiply operation
        self.params.append(np.random.randn(input.shape[1], self.neurons))
        # bias add operation
        self.params.append(np.random.randn(1, self.neurons))
        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation
        ]
