from mnist_dataloader import MNISTDataLoader
from neural_net import NeuralNet
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn
import torch.optim as optim
import torch
import pickle
import compare_metrics as cmp_metrics
from torchvision import transforms
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
start = time.time()



#Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam", "rmsprop"])
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--save_metrics", type=int, default=0)
parser.add_argument("--save_model", type=int, default=0)
args = parser.parse_args()


#Chargement des données via la classe MNSITDataLoader 
loader = MNISTDataLoader()
loader.setup()
train_loader = loader.train_dataloader()
test_loader = loader.test_dataloader()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {device}")


model = NeuralNet.train_model(args.optimizer,loader, train_loader,test_loader, args.epochs, args.save_model, args.save_metrics,device)
end = time.time()
print(f"⏱ Temps total : {end - start:.2f} secondes")

