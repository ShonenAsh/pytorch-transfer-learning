# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# This file trains and tests a Convolutional neural network on the MNIST handwritten digits dataset
# The main function accepts a model_path and mnist_dir_path

import sys
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from torchvision.transforms import functional as TF

from model import MyNetwork


# training a single epoch
def train_single_epoch(epoch, network, optimizer, train_loader, train_counter, train_losses, train_accs, batch_size):
    network.train()
    
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # data = data.to(DEVICE)
        # target = target.to(DEVICE)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.data.max(1, keepdim=True)[1]
        batch_correct = pred.eq(target.data.view_as(pred)).sum().item()
        total_correct += batch_correct
        total_samples += len(data)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}/{} ({:.2f}%))'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),    # Items complete
                    100. * batch_idx / len(train_loader), loss.item(),          # Loss
                    batch_correct, len(data), 100.0 * batch_correct / len(data) # Accuracy
                ))
            train_losses.append(loss.item())
            train_accs.append(100.0 * batch_correct / len(data))
            train_counter.append(
                (batch_idx*batch_size) + ((epoch-1)*len(train_loader.dataset)))


# test the network as described in the tutorial
def test(network, test_loader, test_losses, test_accs):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data = data.to(DEVICE)
            # target = target.to(DEVICE)
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accs.append(100.0 * correct/ len(test_loader.dataset))

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100.0 * correct / len(test_loader.dataset)))



# Trains multiple epochs
def train(train_dataloader, test_dataloader, model_path, epochs=5, lr=0.01, batch_size=128):
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    network = MyNetwork()
    # network = network.to(DEVICE)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.5)

    train_losses = []
    train_accs = []
    train_counter = []
    test_losses = []
    test_accs = []
    test_counter = [(i+1) * len(train_dataloader.dataset) for i in range(epochs)]    

    for t in range(epochs):
        train_single_epoch(t+1, network, optimizer, train_dataloader, train_counter, train_losses, train_accs, batch_size)
        test(network, test_dataloader, test_losses, test_accs)
        print(f"======================")
    print("Complete!")

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')

    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('# training examples seen')
    plt.ylabel('Negative log likelihood loss')
    fig.savefig("./train_test_loss.png")

    fig = plt.figure()
    plt.plot(train_counter, train_accs, color='blue')
    plt.scatter(test_counter, test_accs, color='red')

    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('# training examples seen')
    plt.ylabel('Accuracy')
    fig.savefig("./train_test_acc.png")

    torch.save(network.state_dict(), f"{model_path}/model_{now}.pth")
    torch.save(optimizer.state_dict(), f"{model_path}/optimizer_{now}.pth")
    print(f"Model saved to {model_path}")


def prepare_training_data(path: str, batch_size=128) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
            torchvision.datasets.MNIST(
            path, 
            train=True, 
            download=True, 
            transform=torchvision.transforms.Compose([
                ToTensor(),
                Normalize(
                    (0.1307,),
                    (0.3081,)
                    )]))
            ,batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(
            torchvision.datasets.MNIST(
            path, 
            train=False,
            download=True, 
            transform=torchvision.transforms.Compose([
                ToTensor(),
                Normalize(
                    (0.1307,),
                    (0.3081,)
                    )]))
            ,batch_size=batch_size, shuffle=True)

    print("Test data info:")
    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W] {X.shape}, dtype: {X.dtype}")
        print(f"Shape of y [N] {y.shape}, dtype: {y.dtype}")
        break
    return train_loader, test_loader

# main function, accepts a model_dir and mnist_dir_path
def main(argv):
    if (len(argv) < 3):
        print("Usage: python q1.py <model_dir> <mnist_path>")
        sys.exit(-1)
    train_loader, test_loader = prepare_training_data(argv[2])
    train(train_loader, test_loader, argv[1], epochs=3)
    return 0

if __name__ == "__main__":
    main(sys.argv)
