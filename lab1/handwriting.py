import os
from PIL import Image
import numpy as np
from operation import Linear
from loss import SoftmaxCrossEntropyLoss
from operation import Sigmod
from loss import MeanSquareLoss
from util import onehot_encode
from layer import Dense
from neural_network import Optimizer, Trainer
from operation import Tanh
import torch

from neural_network import NeuralNetwork

DIR = f"{os.path.dirname(__file__)}/train"
X_train = []
y_train = []
X_test = []
y_test = []
for clazz in os.listdir(DIR):
    if os.path.isdir(f"{DIR}/{clazz}"):
        for pic in os.listdir(f"{DIR}/{clazz}"):
            if pic.endswith(".bmp"):
                if np.random.randint(0, 10):
                    X_train.append(
                        list(Image.open(f"{DIR}/{clazz}/{pic}").getdata()))
                    y_train.append(int(clazz)-1)
                else:
                    X_test.append(
                        list(Image.open(f"{DIR}/{clazz}/{pic}").getdata()))
                    y_test.append(int(clazz)-1)

X_train, X_test = np.array(X_train), np.array(X_test)
y_train, y_test = np.array(y_train), np.array(y_test)
t_train, t_test = onehot_encode(y_train, 12), onehot_encode(y_test, 12)


# normalize
X_train = X_train/255.0
X_test = X_test/255.0
# X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
# X_train = X_train / np.std(X_train, axis=1, keepdims=True)
# X_test = X_test - np.mean(X_test, axis=1, keepdims=True)
# X_test = X_test / np.std(X_test, axis=1, keepdims=True)

net = NeuralNetwork([
    Dense(97, Sigmod()),
    Dense(12, Sigmod())
],
    SoftmaxCrossEntropyLoss(),
    seed=1989)

trainer = Trainer(net, Optimizer(0.001))


def validate(X: np.ndarray, y: np.ndarray, net: NeuralNetwork) -> float:
    return (
        np.equal(np.argmax(net.forward(X), axis=1), y).sum()
        * 100.0 / y.shape[0]
    )


trainer.fit(X_train, t_train, X_test, t_test, epochs=50000,
            eval_every=100, seed=202110, batch_size=100)


print(
    f'''
    The model validation accuracy is:
    {validate(X_test,y_test,net)}%
    ''')

torch.save(net, "mynet")
