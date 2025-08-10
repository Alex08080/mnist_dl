import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pickle
from torch.utils.tensorboard import SummaryWriter
import warnings
from sklearn.metrics import confusion_matrix
from datetime import datetime
import seaborn as sns


class NeuralNet(nn.Module) :
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,20,5)  # Premiee couche de convolution 20 sorties pour 1 entrée
        self.bn1 = nn.BatchNorm2d(20)  # Normalise  les activations moyenne à 0 ecart type à 1 pour apprendre plus vite
        self.pool = nn.MaxPool2d(2, 2) # Réduction de la zone de recherche des motifs 
        self.conv2 = nn.Conv2d(20, 50, 5) # 20 entrées 50 sorties
        self.bn2 = nn.BatchNorm2d(50)
        self.conv3 = nn.Conv2d(50, 100, kernel_size=3, padding=1) # 50 entrées 100 sorties
        self.bn3 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(100 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)  # à ajuster selon taille après conv+pool
        self.loss_train_history = []
        self.accuracy_train_history = []
        self.loss_test_history = []
        self.accuracy_test_history = []
        self.dropout = nn.Dropout(p=0.4)
        self.cm = np.zeros((10,10), dtype=int)
 
    def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 100 * 2 * 2)  # aplatissement pour couche fully connected
            x = F.relu(self.fc1(x))  # produit matriciel pour l'avant derniere couche et reLU pour la non linéarité
            x = self.dropout(x)  #appliqué des 0 au hasard parmis les 128 sorties
            x = self.fc2(x) #Dernier couche pour avoir le poids des 10 sorties de 0 à 9
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device : {device}")
        self.eval()
        errors = []
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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

    def confusing_matrix(self, dataloader, device):
        self.eval()
        self.cm = np.zeros((10, 10), dtype=int)
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self(x_batch)
                predicted = y_pred.argmax(dim=1)
                for t, p in zip(y_batch, predicted):
                    self.cm[t.item()][p.item()] += 1
    
    def plot_confusion_matrix(self):
        plt.figure(figsize=(8,6))
        #cm_normalized = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        #sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues")
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Prédiction")
        plt.ylabel("Vérité terrain")
        plt.show()
        accuracy_per_class = self.cm.diagonal() / self.cm.sum(axis=1)
        for i, acc in enumerate(accuracy_per_class):
            print(f"Chiffre {i} - Taux de bonne prédiction : {acc*100:.2f}%")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def accuracy(self, dataloader, device):
        self.eval()  # Mode évaluation
        correct = 0
        total = 0

        with torch.no_grad():  # Pas besoin de calculer les gradients ici
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self(x_batch)
                predicted = y_pred.argmax(dim=1)  # Classe prédite
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = 100 * correct / total
        return accuracy
    
#Entrainement sur un epoch
    def train_epoch(self, train_loader,optimizer, loader,epoch, device,writer):
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = self(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss+= loss.item()
        writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)
        print(f"Epoch {epoch+1}, Loss moyenne : {total_loss/len(train_loader):.4f}")
        self.loss_train_history.append(total_loss/len(train_loader))
        self.accuracy_train_history.append(self.accuracy(train_loader, device))

#Entrainement total
    def train_model(loader, device, args):

        model = NeuralNet().to(device)
        #Creation d'un writer pour ecrire dans le tensorboard
        log_dir = f"../runs/mnist_experiment/best_model_compare/{args.optimizer}_best"
        writer = SummaryWriter(log_dir=log_dir)

        

        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            model_path = "../models/model_sgd_best.pth"
            metrics_path = "../outputs/metrics/metrics_sgd_best.pkl"
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = None
            model_path = "../models/model_adam_base.pth"
            metrics_path = "../outputs/metrics/metrics_adam_base.pkl"
        elif args.optimizer == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
            scheduler = None 
            model_path = "../models/model_rms_base.pth"
            metrics_path = "../outputs/metrics/metrics_rms_base.pkl"
        else:
            raise ValueError(f"Optimizer {args.optimizer} not supported")
        
        

        for epoch in range(args.epochs):
            model.train_epoch(loader.train_dataloader(), optimizer, loader, epoch, device, writer)
            accuracy = model.accuracy(loader.train_dataloader(), device)
            writer.add_scalar("Accuracy/train", accuracy, epoch)
            val_loss = model.evaluate_loss(loader.test_dataloader(), device)
            val_acc = model.accuracy(loader.test_dataloader(), device)
            model.loss_test_history.append(val_loss)
            model.accuracy_test_history.append(val_acc)



            writer.add_scalar("Loss/test", val_loss, epoch)
            writer.add_scalar("Accuracy/test", val_acc, epoch)
            print(f"Epoch {epoch + 1} - Accuracy: {accuracy:.2f}%")
            #if val_acc >= 99.6:
            #    print(f"Converged at epoch {epoch + 1}")
            #    break

            if scheduler is not None:
                scheduler.step()

        if args.plot_confusion:
            model.confusing_matrix(loader.test_dataloader(), device)
            model.plot_confusion_matrix()

        if args.save_model:
            model.save(model_path)

        if args.save_metrics:
            with open(metrics_path, "wb") as f:
                pickle.dump({
                    "loss_train": model.loss_train_history,
                    "accuracy_train": model.accuracy_train_history,
                    "loss_test" : model.loss_test_history,
                    "accuracy_test" : model.accuracy_test_history
                }, f)

        writer.close()
        return model

    def evaluate_loss(self, test_loader, device):
        self.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = self(x_batch)
                loss = loss_fn(y_pred, y_batch)
                total_loss += loss.item()
        return total_loss / len(test_loader)

#Affichage loss et accuracy
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

#Affichage des sorties d'activations
    def plot_output(self,output):
        num_kernels = output.shape[1]
        cols = 6
        rows = (num_kernels + cols - 1) // cols  # arrondi vers le haut

        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(num_kernels):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(output[0, i].detach().cpu().numpy(), cmap='gray')
            plt.title(f'Noyau {i+1}')
            plt.axis('off')
        plt.suptitle("Feature maps après convolution")
        plt.show()
    
#Affichage des sorties d'activations
    def plot_feature_maps(self, feature_maps, title="Feature maps"):
        feature_maps = feature_maps.detach().cpu()

    # Si batch size > 1, on prend le premier exemple
        if feature_maps.dim() == 4:
            feature_maps = feature_maps[0]  # (C, H, W)
        elif feature_maps.dim() == 3:
            pass  # déjà (C, H, W)
        else:
            raise ValueError(f"Format inattendu pour feature_maps: {feature_maps.shape}")

        num_maps = feature_maps.shape[0]
        num_cols = 5
        num_rows = (num_maps + num_cols - 1) // num_cols

        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        for i in range(num_maps):
            fmap = feature_maps[i]
            if fmap.dim() != 2:
                print(f"⚠️ feature_maps[{i}] n'est pas 2D : {fmap.shape}")
                continue  # skip les cas foireux
            plt.subplot(num_rows, num_cols, i + 1)
            plt.imshow(fmap.numpy(), cmap='gray')
            plt.title(f'Noyau {i+1}')
            plt.axis('off')
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.7)
        plt.show()

    def visualize_feature_maps(self, x,device):
        x = x.to(device)
        self.eval()  # mode evaluation pour batchnorm/dropout
        with torch.no_grad():
            self.plot_output(x)
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            self.plot_feature_maps(x)
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            self.plot_feature_maps(x)
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            self.plot_feature_maps(x)

    def plot_conv1_weights(self):
        weights = self.conv1.weight.data.cpu()  # (num_filters, 1, H, W)
        num_filters = weights.shape[0]

        cols = 5
        rows = (num_filters + cols - 1) // cols

        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(num_filters):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(weights[i, 0].numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f'Filtre {i+1}')
        plt.suptitle("Poids des filtres conv1")
        plt.tight_layout()
        plt.show()
    
    def plot_conv2_weights(self):
        weights = self.conv2.weight.data.cpu()  # (50, 20, H, W)
        num_filters = weights.shape[0]

        cols = 5
        rows = (num_filters + cols - 1) // cols

        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(num_filters):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(weights[i, 0].numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f'Filtre {i+1}')
        plt.suptitle("Poids des filtres conv2 (canal 0)")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.7)
        plt.show()

    def plot_conv3_weights(self):
        weights = self.conv3.weight.data.cpu()  # (100, 50, H, W)
        num_filters = weights.shape[0]

        cols = 5
        rows = (num_filters + cols - 1) // cols

        plt.figure(figsize=(cols * 2, rows * 2))
        for i in range(num_filters):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(weights[i, 0].numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f'Filtre {i+1}')
        plt.suptitle("Poids des filtres conv3 (canal 0)")
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, hspace=0.9)
        plt.show()
