from mnist_dataloader import MNISTDataLoader
from neural_net import NeuralNet
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn
import torch.optim as optim
import torch


#Chargement des données via la classe MNSITDataLoader 
loader = MNISTDataLoader()
loader.setup()
train_loader = loader.train_dataloader()

#Chargement du modele grâce à la classe Neural_Net
model = NeuralNet()

#Definition de la loss et de l'optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

'''
#Affichage des données pré entrainement pour voir si on affiche bien les données récupérées
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
'''

#Début de l'entrainement par epochs
for epoch in range(20):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss+= loss.item()
    print(f"Epoch {epoch+1}, Loss moyenne : {total_loss/len(train_loader):.4f}")
    model.eval()  # Mode évaluation
    correct = 0
    total = 0

    with torch.no_grad():  # Pas besoin de calculer les gradients ici
        for x_batch, y_batch in loader.test_dataloader():
            y_pred = model(x_batch)
            predicted = y_pred.argmax(dim=1)  # Classe prédite
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = 100 * correct / total
    print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")

torch.save(model.state_dict(), 'mnist_cnn.pth')

print("\n🔍 Affichage des erreurs de prédiction :")
model.eval()
errors = []

with torch.no_grad():
    for x_batch, y_batch in loader.test_dataloader():
        y_pred = model(x_batch)
        predicted = y_pred.argmax(dim=1)
        for i in range(len(predicted)):
            if predicted[i] != y_batch[i]:
                errors.append((x_batch[i], predicted[i].item(), y_batch[i].item()))
            if len(errors) >= 10:
                break
        if len(errors) >= 10:
            break

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for idx, (img, pred, label) in enumerate(errors):
    ax = axes[idx // 5, idx % 5]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f"Prédit: {pred}\nRéel: {label}", fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.show()


