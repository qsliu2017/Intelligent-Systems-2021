from numpy import ndarray
import numpy as np
import numpy
from numpy.random.mtrand import shuffle
from loss import Loss
from layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss: Loss = Loss()):
        self.layers = layers
        self.loss = loss

    def forward(self, x_batch: ndarray) -> ndarray:
        for layer in self.layers:
            x_batch = layer.forward(x_batch)
        return x_batch

    def backward(self, loss_grad: ndarray):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def train_batch(self, x_batch: ndarray, y_batch: ndarray) -> float:
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads


class Optimizer:
    net: NeuralNetwork

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad


class Trainer:
    def __init__(self, net: NeuralNetwork, optim: Optimizer):
        self.net = net
        self.optim = optim
        optim.net = net

    def generate_batch(self, X: ndarray, Y: ndarray, size: int) -> tuple[ndarray, ndarray]:
        for i in range(0, X.shape[0], size):
            yield X[i:i+size], Y[i:i+size]

    def fit(self,
            X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int = 10000,
            eval_every: int = 10,
            batch_size: int = 32,
            restart: bool = True):
        if restart:
            for layer in self.net.layers:
                layer.first = True
        for e in range(epochs):
            if e % eval_every == 0:  # eval
                eval = self.net.loss.forward(self.net.forward(X_test), y_test)
                print("After %d epochs, loss validation is: %f" %
                      (e, eval/X_test.shape[0]))
            # train
            # np.random.shuffle()
            for x_batch, y_batch in self.generate_batch(X_train, y_train, batch_size):
                self.net.train_batch(x_batch, y_batch)
                self.optim.step()
