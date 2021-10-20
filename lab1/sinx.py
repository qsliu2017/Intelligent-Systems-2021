import matplotlib.pyplot as plt
import numpy
from operation import Tanh
from loss import MeanSquareLoss
from layer import Dense
from operation import Linear
from util import to_2d_np
from neural_network import Optimizer, Trainer, NeuralNetwork

sinx_network = NeuralNetwork([
    Dense(3, Tanh()),
    Dense(1, Linear())
],
    MeanSquareLoss(),
    seed=20211057)


def train_sinx():
    X_train = to_2d_np(numpy.clip(
        numpy.random.randn(100000), -numpy.pi, numpy.pi))
    y_train = numpy.sin(X_train)
    X_test = to_2d_np(numpy.clip(
        numpy.random.randn(1000), -numpy.pi, numpy.pi))
    y_test = numpy.sin(X_test)
    trainner = Trainer(
        sinx_network,
        Optimizer()
    )
    trainner.fit(X_train, y_train,
                 X_test, y_test,
                 seed=20211017,
                 epochs=500)


train_sinx()

x = to_2d_np(numpy.arange(-numpy.pi, numpy.pi, 0.001))
target = numpy.sin(x)
pred = sinx_network.forward(x)

plt.plot(x, target)
plt.plot(x, pred)
plt.show()
