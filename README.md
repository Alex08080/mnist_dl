🧠 Projet MNIST — Semaine 1 : Fondations
Ce dépôt contient les premières étapes d’un projet de classification d’images avec PyTorch, basé sur le dataset MNIST. L’objectif est d’explorer les bases du Deep Learning, de structurer un projet de manière professionnelle, et de suivre sa progression au fil des jours.

## ✅ Jour 1 : Setup et installation
  
### 🔧 Installation de Git + création du dépôt local
### 🐍 Mise en place d’un environnement virtuel (venv) pour éviter les conflits de dépendances 
### 📦 Installation des bibliothèques essentielles :
- numpy
-  matplotlib
-  torch
-  torchvision

### 📚 Lecture de la documentation officielle pour comprendre les concepts suivants :
- PyTorch : Tensors, torch.nn, DataLoader
- Torchvision : datasets et transformations
- Matplotlib : visualisation de données
  
## 🗃️ **Jour 2 : Chargement des données**

### 📥 Dataset chargé via torchvision.datasets.MNIST avec download=True
### ⚙️ Utilisation du DataLoader pour gérer les batches et le shuffle
### 🔍 Test du fonctionnement de next(iter(...)) pour inspecter un batch
### 📊 Visualisation d’exemples de chiffres avec plt.subplots() :
- gestion des axes
- mise en page avec tight_layout()
### 🧱 Début de structuration du code :
- Encapsulation du chargement dans une classe MNISTDataLoader
- Séparation claire des responsabilités
### 🛡️ Création d’un .gitignore adapté :
- fichiers temporaires Python
- fichiers liés à PyTorch
- fichiers système Windows
- modèles sauvegardés (*.pth)
        
## 🧠 Jour 3 : Premier classifieur convolutionnel
### 🧱 Construction d’un modèle NeuralNet :
- Convolution → ReLU → MaxPooling
- Flatten des tensors
- CrossEntropyLoss (intègre log softmax)
### 📐 Compréhension du rôle de chaque opération :
- convolutions pour extraire les features
- Flatten pour passer du 3D au 1D
### 🔁 Mise en place de la boucle d’entraînement :
- Parcours par batch
- Calcul de la loss moyenne
- Backpropagation avec loss.backward()
- Mise à jour des poids avec optimizer.step()
### 🧪 Évaluation du modèle :
- Passage en mode eval()
- Désactivation du calcul des gradients (with torch.no_grad())
- Calcul de l’accuracy via argmax(dim=1) + comparaison avec labels
### 📈 Résultats obtenus :
- 99.14 % d’accuracy avec 2 couches convolutionnelles (8 epochs)
- 99.31 % avec 3 couches convolutionnelles (20 epochs)
### 💾 Sauvegarde du modèle avec torch.save()
  
## 🧠 Jour 4 : Régression linéaire, MLP et visualisation

### 📈 Régression linéaire simple
- Implémentation du calcul des logits :  
  \( z = W \times x + b \)  
  où \( W \) est la matrice de poids, \( x \) le vecteur d’entrée, et \( b \) le biais.  
- Interprétation de \( z \) comme un vecteur de scores pour chaque classe (logits).  
- Initialisation aléatoire des poids et biais, puis prédiction avant entraînement.

### 🧮 Perceptron multicouche (MLP) basique
- Ajout de la fonction d’activation **softmax** pour convertir les logits en probabilités.  
- Implémentation de la fonction **cross_entropy** pour mesurer la perte sur la classe cible.  
- Calcul manuel des gradients, avec mise à jour des poids par descente de gradient.  
- Explication détaillée du gradient `dz` et du mécanisme `dz[target] -= 1`.

### 🔄 Entraînement “from scratch”
- Boucle d’entraînement sur plusieurs epochs avec suivi de la loss.  
- Early stopping manuel lorsque la perte devient suffisamment basse.

### 🧠 Compréhension approfondie de la backpropagation
- Rôle du produit extérieur \( dz[:, np.newaxis] \times x[np.newaxis, :] \) dans le calcul de \( dW \).  
- Impact des gradients sur chaque poids en fonction de la classe cible.

### 🧷 Bonus exploratoire : visualisation des poids
- Reshape de chaque ligne de \( W \) en image 28×28.  
- Visualisation des “patterns” appris par chaque neurone représentant les chiffres typiques.

## ✅ Jour 5 : Optimisation, Réorganisation, Affichage 

### 🧪 Expérimentations de modèles
- Création et entraînement de plusieurs variantes de CNN :
  - [ ] CNN de base sans Dropout ni BatchNorm.
  - [ ] CNN + Dropout uniquement.
  - [ ] CNN + BatchNorm uniquement.
  - [ ] CNN + Dropout + BatchNorm.
- Utilisation de différents optimizers :
  - [ ] SGD
  - [ ] Adam
  - [ ] RMSprop
- Mesure des performances pour chaque combinaison (accuracy max, vitesse de convergence).


### 📊 Performances observées

| Architecture                | Optimizer | Acc. max | Convergence |
|-----------------------------|-----------|----------|-------------|
| CNN                         | SGD       | ~98.8%   | ~15 epochs  |
| CNN + Dropout               | SGD       | ~99.1%   | ~10 epochs  |
| CNN + BatchNorm             | SGD       | 99.3%    | 5 epochs    |
| CNN + Dropout + BatchNorm   | SGD       | **99.36%**| **4 epochs**|

### ⚙️ Techniques approfondies
- **Batch Normalization** :
  - Ajout de `nn.BatchNorm2d` après chaque couche convolutionnelle.
  - Normalisation des activations pour chaque batch.
  - Améliore la stabilité et accélère la convergence.
- **Dropout** :
  - Ajout de `nn.Dropout(p=0.3)` pour régularisation.
  - Appliqué après ReLU et dans les couches fully connected.
- Test de la combinaison BatchNorm + Dropout :
  - Fonctionne bien si les modules sont bien placés.


### 🧠 Compréhensions théoriques
- **BatchNorm** :
  - Réduit l'effet du covariate shift.
  - Rend l'entraînement moins sensible aux initialisations.
  - Permet des taux d'apprentissage plus élevés.
- **Dropout** :
  - Réduction du surapprentissage.
  - Fonctionne comme une régularisation stochastique.
- **Combinaison** :
  - Dropout + BatchNorm fonctionne bien mais doit être positionné intelligemment.


### 🗃️ Organisation du projet
- Réorganisation du code et des sorties dans une arborescence claire 
- Nettoyage des fichiers temporaires.
- Séparation claire des modules : entraînement, visualisation, analyse.


### 📈 Visualisation & Analyse
- Sauvegarde automatique des métriques (`loss`, `accuracy`) dans des fichiers `.pkl`.
- Comparaison visuelle via des courbes matplotlib.
- Export des figures sous forme d'images `.png`.

