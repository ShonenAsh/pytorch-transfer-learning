# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# This file trains and tests a Convolutional neural network on the MNIST handwritten digits dataset

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import datasets
import torchvision
from torchvision.transforms import Normalize, ToTensor
from typing import Tuple
import sys
from datetime import datetime

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.functional.rgb_to_grayscale(x)
        x = torchvision.torchvision.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.functional.center_crop(x, (28, 28))
        x = torchvision.functional.invert(x)
        return x

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.maxpool1(x))
        x = self.conv2(x)
        x = F.dropout2d(x, 0.1)
        x = F.relu(self.maxpool2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)

        return x

# trains a single epoch, going over the entire dataset in batches.
def train_single_epoch(train_dataloader: DataLoader, network: nn.Module, optimizer, device):
    size = len(train_dataloader.dataset)
    network.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        pred = network(X)
        loss = F.nll_loss(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Trains multiple epochs
def train(train_dataloader, test_dataloader, model_path="./model", epochs=5, lr=0.01, device="cpu"):
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    network = MyNetwork().to(device)
    print(network)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.5)

    for t in range(epochs):
        print(f"===== Epoch: {t+1} =====")
        train_single_epoch(train_dataloader, network, optimizer, device)
        test(test_dataloader, network, device)
        print(f"======================")
    print("Complete!")
    torch.save(network.state_dict(), f"{model_path}/model_{now}.pth")
    torch.save(optimizer.state_dict(), f"{model_path}/optimizer_{now}.pth")
    print(f"Model saved to {model_path}")


def test(dataloader, model, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += F.nll_loss(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches

    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def prepare_training_data(path: str, batch_size=128) -> Tuple[DataLoader, DataLoader]:
    mnist_train_data = datasets.MNIST(
            path, 
            train=True, 
            download=True, 
            transform=torchvision.transforms.Compose([
                ToTensor(),
                Normalize(
                    (0.1307,),
                    (0.3081,)
                    )]))
    mnist_test_data = datasets.MNIST(
            path, 
            train=False, 
            download=True, 
            transform=torchvision.transforms.Compose([
                ToTensor(),
                Normalize(
                    (0.1307,),
                    (0.3081,)
                    )]))
    train_dataloader = DataLoader(mnist_train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test_data, batch_size=batch_size, shuffle=True)
    print("Test data info:")
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W] {X.shape}, dtype: {X.dtype}")
        print(f"Shape of y [N] {y.shape}, dtype: {y.dtype}")
        break
    return train_dataloader, test_dataloader

# main function (yes, it needs a comment too)
def main(argv):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    train_loader, test_loader = prepare_training_data("./data/")
    train(train_loader, test_loader, epochs=15, device=DEVICE)
    return 0

if __name__ == "__main__":
    main(sys.argv)
