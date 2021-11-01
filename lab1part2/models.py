from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.net.forward(x)


class Extern_LeNet5(nn.Module):
    def __init__(self):
        super(Extern_LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.net.forward(x)


class Small_LeNet5(nn.Module):
    def __init__(self):
        super(Small_LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2304, 25600),
            nn.ReLU(),
            nn.Linear(25600, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.net.forward(x)


class Zero_LeNet5(nn.Module):
    def __init__(self):
        super(Zero_LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 80),
            nn.ReLU(),
            nn.Linear(80, 25600),
            nn.ReLU(),
            nn.Linear(25600, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.net.forward(x)


class Dropout_LeNet5(nn.Module):
    def __init__(self, dropout_p: float):
        super(Dropout_LeNet5, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(dropout_p),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, x):
        return self.net.forward(x)
