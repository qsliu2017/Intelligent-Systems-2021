import os

from PIL import Image
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import Lambda


class HandWritingDataset(Dataset):
    def __init__(self, dir: str) -> None:
        self.dir = dir
        self.transform = ToTensor()
        self.target_transform = Lambda(lambda y: y-1)

        # [(path, lable)]
        self.paths: list[tuple[str, str]] = [
            (image, lable)
            for lable in os.listdir(dir)
            if os.path.isdir(os.path.join(dir, lable))
            for image in os.listdir(os.path.join(dir, lable))
            if image.endswith(".bmp")
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path, lable = self.paths[index]
        return (
            self.transform(Image.open(os.path.join(self.dir, lable, path))),
            self.target_transform(int(lable))
        )


handWritingDataset = HandWritingDataset("lab1part2/train")


def divide_dataset(factor: float) -> tuple[Dataset, Dataset]:
    test_size = int(factor*len(handWritingDataset))
    train_size = len(handWritingDataset) - test_size
    return random_split(handWritingDataset, [train_size, test_size])


def divide_dataloader(factor: float = .1, batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    train_dataset, test_dataset = divide_dataset(factor)
    return (
        DataLoader(train_dataset, batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size)
    )
