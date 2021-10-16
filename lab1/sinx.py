import matplotlib.pyplot as plt
import numpy
from loss import Loss
from operation import Sigmod, Linear
from util import to_2d_np
from layer import Layer
from neural_network import Optimizer, Trainer, NeuralNetwork

sinx_network = NeuralNetwork([
    Layer(3, Sigmod()),
    Layer(1, Linear())
],
    Loss(),
    seed=20211057)


def train_sinx():
    X_train = to_2d_np(numpy.arange(-numpy.pi, numpy.pi, 0.0001))
    y_train = numpy.sin(X_train)
    X_test = to_2d_np(numpy.arange(-numpy.pi, numpy.pi, 0.001))
    y_test = numpy.sin(X_test)
    trainner = Trainer(
        sinx_network,
        Optimizer()
    )
    trainner.fit(X_train, y_train, X_test, y_test, seed=20211017)


train_sinx()

x = to_2d_np(numpy.arange(-numpy.pi, numpy.pi, 0.001))
target = numpy.sin(x)
pred = sinx_network.forward(x)

plt.plot(x, target)
plt.plot(x, pred)
plt.show()
