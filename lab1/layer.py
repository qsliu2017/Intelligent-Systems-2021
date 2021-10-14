import time
import numpy as np
from numpy import ndarray
from operation import Operation, ParamOperation, Sigmod, BiasAdd, WeightMultiply


class Layer(object):
    def __init__(self, neurons: int,
                 activation: Operation = Sigmod(), seed=int(time.time())):
        self.neurons = neurons
        self.activation = activation
        self.seed = seed
        self.first = True
        self.params: list[ndarray] = []
        self.param_grads: list[ndarray] = []
        self.operations: list[Operation] = []

    def _setup_layer(self, input: ndarray):
        np.random.seed(self.seed)
        self.params = []
        # weight multiply operation
        self.params.append(np.random.standard_cauchy(
            (input.shape[1], self.neurons))/self.neurons)
        # bias add operation
        self.params.append(-np.abs(np.random.standard_cauchy((1, self.neurons))))
        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation
        ]

    def forward(self, input: ndarray) -> ndarray:
        if self.first:
            self._setup_layer(input)
            self.first = False
        for operation in self.operations:
            input = operation.forward(input)
        self.output = input
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        self._param_grads()
        return output_grad

    def _param_grads(self) -> ndarray:
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad)

    def _params(self) -> ndarray:
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
