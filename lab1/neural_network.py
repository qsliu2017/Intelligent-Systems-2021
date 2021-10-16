from numpy import ndarray
import numpy as np
from loss import Loss
from layer import Layer


class NeuralNetwork:
    def __init__(self, layers: list[Layer], loss: Loss, seed: int = 1):
        self.layers = layers
        self.loss = loss
        self.seed = seed
        if seed:
            for layer in self.layers:
                setattr(layer, "seed", self.seed)

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
        setattr(self.optim, 'net', self.net)

    def generate_batch(self, X: ndarray, y: ndarray, size: int = 32) -> tuple[ndarray, ndarray]:
        assert X.shape[0] == y.shape[0], \
            '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(X.shape[0], y.shape[0])
        for i in range(0, X.shape[0], size):
            X_batch, y_batch = X[i:i+size], y[i:i+size]
            yield X_batch, y_batch

    def shuffle_batch(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray]:
        perm = np.random.permutation(X.shape[0])
        return X[perm], y[perm]

    def fit(self,
            X_train: ndarray, y_train: ndarray,
            X_test: ndarray, y_test: ndarray,
            epochs: int = 100,
            eval_every: int = 10,
            batch_size: int = 32,
            seed: int = 1,
            restart: bool = True):
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
        for e in range(epochs):
            if (e+1) % eval_every == 0:  # eval
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                print(f"After {e+1} epochs, loss validation is: {loss}")
            # train
            X_train, y_train = self.shuffle_batch(X_train, y_train)
            for X_batch, y_batch in self.generate_batch(X_train, y_train, batch_size):
                self.net.train_batch(X_batch, y_batch)
                self.optim.step()
