import time
import numpy
from layer import Layer
from neural_network import Optimizer, Trainer, NeuralNetwork


def train_sinx():
    rng = numpy.random.default_rng()
    y_train = rng.random((1000, 1), numpy.float32)
    X_train = numpy.arcsin(y_train)
    y_test = rng.random((100, 1), numpy.float32)
    X_test = numpy.arcsin(y_test)
    trainner = Trainer(
        NeuralNetwork([
            Layer(1),
            Layer(3),
            Layer(1)
        ]),
        Optimizer(lr=0.01)
    )
    trainner.fit(X_train, y_train, X_test, y_test)


train_sinx()
