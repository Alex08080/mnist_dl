🧠 Projet MNIST — Semaine 1 : Fondations
Ce dépôt contient les premières étapes d’un projet de classification d’images avec PyTorch, basé sur le dataset MNIST. L’objectif est d’explorer les bases du Deep Learning, de structurer un projet de manière professionnelle, et de suivre sa progression au fil des jours.

✅ **Jour 1 : Setup et installation**
  
  🔧 Installation de Git + création du dépôt local
  
  🐍 Mise en place d’un environnement virtuel (venv) pour éviter les conflits de dépendances 
  
  📦 Installation des bibliothèques essentielles :
        numpy
        
        matplotlib
        
        torch
        
        torchvision
        
  📚 Lecture de la documentation officielle pour comprendre les concepts suivants :
  
        PyTorch : Tensors, torch.nn, DataLoader
        
        Torchvision : datasets et transformations
        
        Matplotlib : visualisation de données
  
🗃️ **Jour 2 : Chargement des données**

  📥 Dataset chargé via torchvision.datasets.MNIST avec download=True
  
  ⚙️ Utilisation du DataLoader pour gérer les batches et le shuffle
  
  🔍 Test du fonctionnement de next(iter(...)) pour inspecter un batch
  
  📊 Visualisation d’exemples de chiffres avec plt.subplots() :
  
        gestion des axes
        
        mise en page avec tight_layout()
        
  🧱 Début de structuration du code :
  
        Encapsulation du chargement dans une classe MNISTDataLoader
        
        Séparation claire des responsabilités

  🛡️ Création d’un .gitignore adapté :
  
        fichiers temporaires Python
        
        fichiers liés à PyTorch
        
        fichiers système Windows
        
        modèles sauvegardés (*.pth)
        
🧠 **Jour 3 : Premier classifieur convolutionnel**

  🧱 Construction d’un modèle NeuralNet :
  
        Convolution → ReLU → MaxPooling
        
        Flatten des tensors
        
        CrossEntropyLoss (intègre log softmax)
        
  📐 Compréhension du rôle de chaque opération :
  
        convolutions pour extraire les features
        
        Flatten pour passer du 3D au 1D
        
  🔁 Mise en place de la boucle d’entraînement :
  
        Parcours par batch
        
        Calcul de la loss moyenne
        
        Backpropagation avec loss.backward()
        
        Mise à jour des poids avec optimizer.step()
        
  🧪 Évaluation du modèle :
  
        Passage en mode eval()
        
        Désactivation du calcul des gradients (with torch.no_grad())
        
        Calcul de l’accuracy via argmax(dim=1) + comparaison avec labels
        
  📈 Résultats obtenus :
  
        99.14 % d’accuracy avec 2 couches convolutionnelles (8 epochs)
        
        99.31 % avec 3 couches convolutionnelles (20 epochs)
        
  💾 Sauvegarde du modèle avec torch.save()
