from mnist_dataloader import MNISTDataLoader
from neural_net import NeuralNet
import matplotlib.pyplot as plt # type: ignore
import torch.nn as nn
import torch.optim as optim
import torch
import pickle
import compare_metrics as cmp_metrics


#Chargement des données via la classe MNSITDataLoader 
loader = MNISTDataLoader()
loader.setup()
train_loader = loader.train_dataloader()

#Chargement du modele grâce à la classe Neural_Net
model_rms = NeuralNet()
model_sgd = NeuralNet()
model_adam = NeuralNet()

#Definition de la loss et de l'optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.0001)
optimizer_rms = optim.RMSprop(model_rms.parameters(), lr=0.0001)


'''
#Entrainement RMS
for epoch in range(20):
    model_rms.train_epoch(train_loader, optimizer_rms, loader, epoch)
    accuracy = model_rms.accuracy(loader)
    if accuracy >= 99.3:
        print(f"Converged at epoch {epoch}")
        print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")
        break
    print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")

with open("metrics/metrics_rms.pkl", "wb") as f:
    pickle.dump({
        "loss": model_rms.loss_history,
        "accuracy": model_rms.accuracy_history
    }, f)

'''

#Entrainement SGD

for epoch in range(20):
    model_sgd.train_epoch(train_loader, optimizer_sgd, loader, epoch)
    accuracy = model_sgd.accuracy(loader)
    '''
    if accuracy >= 99.3:
        print(f"Converged at epoch {epoch}")
        print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")
        break
    '''
    print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")
'''
with open("metrics/metrics_sgd_with_batchnorm_and_dropout.pkl", "wb") as f:
    pickle.dump({
        "loss": model_sgd.loss_history,
        "accuracy": model_sgd.accuracy_history
    }, f)
'''

'''

#Entrainement ADAM

for epoch in range(20):
    model_adam.train_epoch(train_loader, optimizer_adam, loader, epoch)
    accuracy = model_adam.accuracy(loader)
    if accuracy >= 99.3:
        print(f"Converged at epoch {epoch}")
        print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")
        break
    print(f"🧪 Accuracy sur le test set : {accuracy:.2f}%")

with open("metrics/metrics_adam.pkl", "wb") as f:
    pickle.dump({
        "loss": model_adam.loss_history,
        "accuracy": model_adam.accuracy_history
    }, f)
'''

#cmp_metrics.plot_loss_and_accuracy()