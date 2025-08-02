from torchvision import datasets 
from torchvision.transforms import ToTensor # type: ignore
from torch.utils.data import DataLoader # type: ignore


class MNISTDataLoader:
    def __init__(self, data_dir="data", batch_size=64):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = datasets.MNIST(root=self.data_dir, train=True, transform=ToTensor(), download=True)
        self.test_dataset = datasets.MNIST(root=self.data_dir, train=False, transform=ToTensor(), download=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    

mnsit_dataset = datasets.MNIST(root = "data",train = False, transform = ToTensor(),download = True)
mnist_test = DataLoader(mnsit_dataset, batch_size = 64, shuffle = True)
