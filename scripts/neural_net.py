import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(100 * 2 * 2, 128)  # à ajuster selon taille après conv+pool
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 100 * 2 * 2)  # aplatissement pour couche fully connected
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
