import copy

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from dataset import divide_dataloader
from models import Extern_LeNet5, LeNet5, Small_LeNet5, Zero_LeNet5


def validate(model: nn.Module, dataloader: DataLoader, loss_fn: _Loss) -> tuple[float, float]:
    '''
    (accuracy, loss)
    '''
    correct, loss = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return (correct / len(dataloader.dataset), loss / len(dataloader))


class Trainer:
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 model: nn.Module, loss_fn: _Loss, optimizer: Optimizer) -> None:
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.test_size = len(self.test_dataloader.dataset)
        self.num_test_batcher = len(self.test_dataloader)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    # return accuracy and loss in train set
    def train_loop(self) -> tuple[float]:
        for X, y in self.train_dataloader:
            pred = self.model(X)
            self.optimizer.zero_grad()
            self.loss_fn(pred, y).backward()
            self.optimizer.step()
        return validate(self.model, self.train_dataloader, self.loss_fn)

    def test_loop(self) -> tuple[float, float]:
        return validate(self.model, self.test_dataloader, self.loss_fn)

    def train(self, epochs: int = 50, eval_every: int = 1, save_best: bool = False) -> tuple[list[float], list[float], list[float], list[float]]:
        '''
        (trainset_accuracy, trainset_loss, testset_accuracy, testset_loss)
        '''
        best_correct = .0
        best_model = copy.deepcopy(self.model)

        trainset_accuracy, trainset_loss, testset_accuracy, testset_loss = [], [], [], []
        for e in range(epochs):
            accuracy, loss = self.train_loop()
            trainset_accuracy.append(accuracy)
            trainset_loss.append(loss)

            if (e+1) % eval_every == 0:
                accuracy, loss = self.test_loop()
                testset_accuracy.append(accuracy)
                testset_loss.append(loss)
                print(
                    f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg Loss: {loss:>8f}\n")
                if save_best and accuracy > best_correct:
                    best_model = copy.deepcopy(self.model)
                    best_correct = accuracy

        if save_best:
            print("Saving model:", best_model,
                  "with best correct:", best_correct)
            torch.save(best_model, f"BestModel{best_correct}")

        return (trainset_accuracy, trainset_loss, testset_accuracy, testset_loss)

