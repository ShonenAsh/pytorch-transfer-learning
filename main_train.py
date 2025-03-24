# Ashish Magadum & Nicholas Payson
# This file trains a neural network

import torch
from torch import device, nn
from torch.utils.data import DataLoader 
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Tuple
import sys


# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack= nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
                )
    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# useful functions with a comment for each function
def train_network(train_dataloader: DataLoader, network: nn.Module, lr=1e-3, device='cpu'):
    print(network)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)

    size = len(train_dataloader.dataset)
    network.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        pred = network(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    return

def test_network(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches

    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def prepare_training_data(path: str, batch_size=1024) -> Tuple[DataLoader, DataLoader]:
    mnist_train_data = datasets.MNIST(path, train=True, download=True, transform=ToTensor())
    mnist_test_data = datasets.MNIST(path, train=False, download=True, transform=ToTensor())
    train_dataloader = DataLoader(mnist_train_data, batch_size=batch_size)
    test_dataloader = DataLoader(mnist_test_data, batch_size=batch_size)
    print("Test data info:")
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W] {X.shape}, dtype: {X.dtype}")
        print(f"Shape of y [N] {y.shape}, dtype: {y.dtype}")
        break
    return train_dataloader, test_dataloader

# main function (yes, it needs a comment too)
def main(argv):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = prepare_training_data("./data/")
    model = MyNetwork().to(DEVICE)
    train_network(train_loader, model, device=DEVICE)
    return 0

if __name__ == "__main__":
    main(sys.argv)
