from mnist_dataloader import MNISTDataLoader
import matplotlib.pyplot as plt # type: ignore

loader = MNISTDataLoader()
loader.setup()
train_loader = loader.train_dataloader()

train_features, train_labels = next(iter(train_loader))

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