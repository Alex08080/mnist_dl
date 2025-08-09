from torchvision import datasets 
from torchvision.transforms import ToTensor # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision import transforms

def train_transform():
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return train_transform

def test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return test_transform

class MNISTDataLoader:
    def __init__(self, data_dir="data", batch_size=128):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=train_transform(), download=True)
        self.test_dataset = datasets.MNIST(root=self.data_dir, train=False, transform=test_transform(), download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory = True)


