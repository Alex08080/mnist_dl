import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    z = z - np.max(z) #evite les nombres tres grand vu qu'on utilse des exponentielles
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z) # nous permet d'avoir le poids de chaque classe de sortie pour savoir quelle classe le modèle prédit

#Fonction qui la loss avec la méthode de cross_entropy
def cross_entropy(y_pred, target_index):
    return -np.log(y_pred[target_index])

 # vecteur d'entrée
x = np.random.randn(784) 

# Poids de la couche linéaire : 10 sorties (1 par chiffre)
W = np.random.randn(10, 784) * 0.01
b = np.zeros(10)

# Classe cible (ex : chiffre 3)
target = 3

lr = 0.01
epochs = 10
losses = []

for epoch in range(epochs):
    z = W @ x + b   
    y_pred = softmax(z)

    loss = cross_entropy(y_pred, target)
    losses.append(loss)
    #Calcul des gradients
    dz = y_pred.copy()
    dz[target]-=1
    dW = dz[:, np.newaxis] @ x[np.newaxis, :]  # (10, 784)
    db = dz

    # maj des poids

    W -= lr * dW
    b -= lr * db

    print("z (logits) :")
    print(z.round(3))
    print("\nsoftmax(y_pred) :")
    print(y_pred.round(3))
    print("\ndz (gradient de la perte par rapport à z) :")
    print(dz.round(3))
    print(f"Epoch {epoch}: loss = {loss:.4f}")
    print(f"W mean = {W.mean():.4f}, b mean = {b.mean():.4f}")
    if loss < 1e-4:
        print(f"Converged at epoch {epoch}")
        break

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence")
plt.grid(True)
plt.show()