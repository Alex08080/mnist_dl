import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Dataset MNIST (juste un mini batch pour exemple)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 2. Modèle simple linéaire
model = torch.nn.Linear(784, 10)

# 3. Optimizer et loss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. Entraînement simple sur quelques batchs
for epoch in range(15):
    for inputs, targets in trainloader:
        inputs = inputs.view(-1, 784)  # aplatissement 28x28 -> 784
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# 5. Récupérer la matrice de poids et afficher les images
weights = model.weight.data  # shape (10, 784)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = weights[i].reshape(28, 28).cpu().numpy()
    ax.imshow(img, cmap='seismic')  # cmap séisme pour voir positif/négatif
    ax.set_title(f'Classe {i}')
    ax.axis('off')

plt.show()
