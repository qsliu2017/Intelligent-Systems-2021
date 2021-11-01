from matplotlib import pyplot as plt
from torch import nn, optim
import torch

from dataset import divide_dataloader
from models import Dropout_LeNet5
from trainer import Trainer

trainDataloader, testDataloader = divide_dataloader(0.9)

learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()

loss_axes: plt.Axes = plt.subplot(2, 1, 1)
accuracy_axes: plt.Axes = plt.subplot(2, 1, 2, sharex=loss_axes)

for dropout in torch.range(0.0, 0.2, 0.05):
    model = Dropout_LeNet5(dropout)
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate)
    trainer = Trainer(trainDataloader, testDataloader,
                      model, loss_fn, optimizer)
    _, _, testset_accuracy, testset_loss = trainer.train(
        100)

    accuracy_axes.plot(testset_accuracy, label=f"{dropout:.2f}")
    loss_axes.plot(testset_loss, label=f"{dropout:.2f}")

accuracy_axes.legend()
loss_axes.legend()

accuracy_axes.set_title("Accuracy")
loss_axes.set_title("Loss")
plt.tight_layout()
plt.show()
