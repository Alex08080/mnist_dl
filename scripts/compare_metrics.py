import matplotlib.pyplot as plt
import pickle

def load_metrics(path):
    with open(path, "rb") as f:
        return pickle.load(f)


optimizers = ["sgd_best","sgd_wo_batchnorm","sgd_wo_dropout","sgd_wo_data_augment"]
colors = ["blue","orange","purple","red"]


#  Courbe de loss_train
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
for opti, color in zip(optimizers, colors):
    data = load_metrics(f"../outputs/metrics/metrics_{opti.lower()}.pkl")
    plt.plot(data["loss_train"], label=f"{opti}", color=color)
plt.title("Comparaison des pertes (loss_train)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

#  Courbe d'accuracy_train
plt.subplot(2, 2, 2)
for opti, color in zip(optimizers, colors):
    data = load_metrics(f"../outputs/metrics/metrics_{opti.lower()}.pkl")
    plt.plot(data["accuracy_train"], label=f"{opti}", color=color)
plt.title("Comparaison des précisions (accuracy_train)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

#  Courbe de loss_test
plt.subplot(2, 2, 3)
for opti, color in zip(optimizers, colors):
    data = load_metrics(f"../outputs/metrics/metrics_{opti.lower()}.pkl")
    plt.plot(data["loss_test"], label=f"{opti}", color=color)
plt.title("Comparaison des pertes (loss_test)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

#  Courbe d'accuracy_test
plt.subplot(2, 2, 4)
for opti, color in zip(optimizers, colors):
    data = load_metrics(f"../outputs/metrics/metrics_{opti.lower()}.pkl")
    plt.plot(data["accuracy_test"], label=f"{opti}", color=color)
plt.title("Comparaison des précisions (accuracy_test)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("../outputs/plots/best_compares.png")
plt.show()