from matplotlib import pyplot as plt
from torch import nn, optim

from dataset import divide_dataloader
from models import LeNet5, Small_LeNet5, Zero_LeNet5
from trainer import Trainer

trainDataloader, testDataloader = divide_dataloader(0.3)

learning_rate = 1e-2
loss_fn = nn.CrossEntropyLoss()

loss_axes: plt.Axes = plt.subplot(2, 1, 1)
accuracy_axes: plt.Axes = plt.subplot(2, 1, 2, sharex=loss_axes)

model = Zero_LeNet5()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
trainer = Trainer(trainDataloader, testDataloader, model, loss_fn, optimizer)
_, _, testset_accuracy, testset_loss = trainer.train(
    50)

accuracy_axes.plot(testset_accuracy, label="zero-LeNet5")
loss_axes.plot(testset_loss, label="zero-LeNet5")

# model = Extern_LeNet5()
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
# trainer = Trainer(trainDataloader, testDataloader, model, loss_fn, optimizer)
# _, _, testset_accuracy, testset_loss = trainer.train(
#     50)

# plt.plot(testset_accuracy, label="accuracy of extern-LeNet5")
# plt.plot(testset_loss, label="loss of extern-LeNet5")

model = LeNet5()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
trainer = Trainer(trainDataloader, testDataloader, model, loss_fn, optimizer)
_, _, testset_accuracy, testset_loss = trainer.train(
    50)

accuracy_axes.plot(testset_accuracy, label="LeNet5")
loss_axes.plot(testset_loss, label="LeNet5")

model = Small_LeNet5()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
trainer = Trainer(trainDataloader, testDataloader, model, loss_fn, optimizer)
_, _, testset_accuracy, testset_loss = trainer.train(
    50)

accuracy_axes.plot(testset_accuracy, label="small-LeNet5")
loss_axes.plot(testset_loss, label="small-LeNet5")

accuracy_axes.legend()
loss_axes.legend()

accuracy_axes.set_title("Accuracy")
loss_axes.set_title("Loss")
plt.tight_layout()
plt.show()
