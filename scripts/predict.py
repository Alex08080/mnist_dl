import torch
from torchvision import transforms
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
from neural_net import NeuralNet

model = NeuralNet()
model.load("../models/mnist_cnn_sgd_batchnorm_dropout.pth")
model.eval()

transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

img = Image.open("../test/neuf.png")
img = ImageOps.invert(img)
img = transform(img).unsqueeze(0)

with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)
        print(f"Prédiction : {pred.item()}")
        print(torch.softmax(output, dim=1))