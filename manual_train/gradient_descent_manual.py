import numpy as np
import matplotlib.pyplot as plt

#initilisation des données de base
x = np.arange(1001)
x = (x - np.mean(x)) / np.std(x)
y = 2 * x + 1

#Intilisations des parametres que l'on veut atteindre
w = 9
b = -10

#Initialisation des hyperparamètre
lr = 0.4
epochs = 10000
losses = []

for epoch in range(epochs):
    y_pred = w * x + b
    loss = np.mean((y_pred - y)**2)
    losses.append(loss)
    #Calcul des gradients
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 *(y_pred - y))

    # maj des poids

    w -= lr * dw
    b -= lr * db

    print(f"Epoch {epoch}: loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
    if loss < 1e-4:
        print(f"Converged at epoch {epoch}")
        break


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Évolution de la loss")
plt.grid()
plt.show()

