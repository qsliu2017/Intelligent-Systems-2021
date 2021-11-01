from torch import nn, optim
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from dataset import divide_dataloader
from dataset import HandWritingDataset
from trainer import validate
from models import LeNet5
from trainer import Trainer


def train():
    trainDataloader, testDataloader = divide_dataloader(0.3)

    learning_rate = 1e-3
    loss_fn = nn.CrossEntropyLoss()
    model = LeNet5()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=.9)
    trainer = Trainer(trainDataloader, testDataloader,
                      model, loss_fn, optimizer)
    trainer.train(100, save_best=True)


def _validate():
    model = torch.load("BestModel0.9780465949820788")
    validate_dataloader = DataLoader(HandWritingDataset("lab1part2/train"))
    accuracy, loss = validate(model, validate_dataloader, CrossEntropyLoss())
    print(f"accuracy:{accuracy*100.0:.2f}% Loss:{loss:.2f}")


if __name__ == "__main__":
    _validate()
