import matplotlib.pyplot as plt
import pickle

def load_metrics(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def plot_loss_and_accuracy():
    optimizers = ["sgd", "sgd_with_dropout","sgd_with_batchnorm", "sgd_with_batchnorm_and_dropout"]
    colors = ["blue", "green", "red", "orange"]


#  Courbe de loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opti, color in zip(optimizers, colors):
        data = load_metrics(f"metrics/metrics_{opti.lower()}.pkl")
        plt.plot(data["loss"], label=f"{opti}", color=color)
    plt.title("Comparaison des pertes (loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

#  Courbe d'accuracy
    plt.subplot(1, 2, 2)
    for opti, color in zip(optimizers, colors):
        data = load_metrics(f"metrics/metrics_{opti.lower()}.pkl")
        plt.plot(data["accuracy"], label=f"{opti}", color=color)
    plt.title("Comparaison des précisions (accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("plots/comparaison_loss_accuracy_all_sgd.png")
    plt.show()