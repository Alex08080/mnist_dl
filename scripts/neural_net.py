import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

class NeuralNet(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.bn1 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(100 * 2 * 2, 128)  # à ajuster selon taille après conv+pool
        self.fc2 = nn.Linear(128, 10)
        self.loss_history = []
        self.accuracy_history = []
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 100 * 2 * 2)  # aplatissement pour couche fully connected
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
#Affichage des données pré entrainement pour voir si on affiche bien les données récupérées
    def show_batch(self, features, labels, n_rows=5, n_cols=6):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 8))
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                img = features[idx].squeeze()
                label = labels[idx]
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].set_title(str(label), fontsize=8)
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.show()

#Affichage de quelques erreurs du modèle pour visualiser les images qu'il n'arrive pas à bien prédire 
    def show_misclassified(self, dataloader, max_errors=10):
        print("\n🔍 Affichage des erreurs de prédiction :")
        self.eval()
        errors = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                y_pred = self(x_batch)
                predicted = y_pred.argmax(dim=1)
                for i in range(len(predicted)):
                    if predicted[i] != y_batch[i]:
                        errors.append((x_batch[i], predicted[i].item(), y_batch[i].item()))
                    if len(errors) >= max_errors:
                        break
                if len(errors) >= max_errors:
                    break
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for idx, (img, pred, label) in enumerate(errors):
            ax = axes[idx // 5, idx % 5]
            ax.imshow(img.squeeze(), cmap='gray')
            ax.set_title(f"Prédit: {pred}\nRéel: {label}", fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def accuracy(self, loader):
        self.eval()  # Mode évaluation
        correct = 0
        total = 0

        with torch.no_grad():  # Pas besoin de calculer les gradients ici
            for x_batch, y_batch in loader.test_dataloader():
                y_pred = self(x_batch)
                predicted = y_pred.argmax(dim=1)  # Classe prédite
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = 100 * correct / total
        return accuracy
    
    def train_epoch(self, train_loader,optimizer, loader,epoch):
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = self(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss+= loss.item()
        print(f"Epoch {epoch+1}, Loss moyenne : {total_loss/len(train_loader):.4f}")
        self.loss_history.append(total_loss/len(train_loader))
        self.accuracy_history.append(self.accuracy(loader))

    def plot_metrics(self, optimizer_name):
        epochs = range(1, len(self.loss_history) +1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1,2,1)
        plt.plot(epochs, self.loss_history, label = f"Loss - {optimizer_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Evolution de la loss")
        plt.grid(True)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, self.accuracy_history, label = f"Loss - {optimizer_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Evolution de l'accuracy ")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    