import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader # type: ignore

mnsit_dataset = datasets.MNIST(root = "data",train = False, transform = ToTensor(),download = True)
mnist_test = DataLoader(mnsit_dataset, batch_size = 64, shuffle = True)

train_features, train_labels = next(iter(mnist_test))
fig, axes = plt.subplots(5, 6, figsize=(10, 8))
for i in range(5):
    for j in range(6):
        idx = i * 6 + j
        img = train_features[idx].squeeze()
        label = train_labels[idx]
        axes[i, j].imshow(img, cmap='gray')
        axes[i, j].set_title(str(label), fontsize=8)
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
