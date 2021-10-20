
import os

from PIL import Image
import numpy as np
from neural_network import NeuralNetwork
from util import to_2d_np

import torch


DIR = "lab1/train"
X = []
y = []
for clazz in os.listdir(DIR):
    if os.path.isdir(f"{DIR}/{clazz}"):
        for pic in os.listdir(f"{DIR}/{clazz}"):
            if pic.endswith(".bmp"):
                X.append(
                    list(Image.open(f"{DIR}/{clazz}/{pic}").getdata()))
                y.append(int(clazz)-1)

X = np.array(X)
y = np.array(y)


# normalize
X = X/255.0


def validate(X: np.ndarray, y: np.ndarray, net: NeuralNetwork) -> float:
    return (
        np.equal(np.argmax(net.forward(X), axis=1), y).sum()
        * 100.0 / y.shape[0]
    )


net: NeuralNetwork = torch.load("mynet")


print(
    f'''
The model validation accuracy is:
{validate(X,y,net)}%
''')
