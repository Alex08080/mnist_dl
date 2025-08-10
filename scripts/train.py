from mnist_dataloader import MNISTDataLoader
from neural_net import NeuralNet
import torch
import argparse
import time
start = time.time()



#Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "rmsprop"])
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--save_metrics", type=int, default=0)
parser.add_argument("--save_model", type=int, default=0)
parser.add_argument("--plot_confusion", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.01)


args = parser.parse_args()


#Chargement des données via la classe MNSITDataLoader 
loader = MNISTDataLoader()
loader.setup()
train_loader = loader.train_dataloader()
test_loader = loader.test_dataloader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")


model = NeuralNet.train_model(loader,device, args)
end = time.time()
print(f"⏱ Temps total : {end - start:.2f} secondes")

