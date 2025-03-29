# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# This file retrains an existing model file and tests it on greek letters
# The main function, accepts a model pth file, greek_train_dir, greek_test_dir

from ast import arg
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

# Apply grayscale, random rotations and center cropping followed by invert
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = TF.rgb_to_grayscale( x )
        x = TF.affine( x, 0, (0,0), 36/128, 0 )
        x = TF.center_crop(x, output_size=(28, 28))
        return TF.invert( x )


# train the network as described in the tutorial
def train_single_epoch(epoch, network, optimizer, train_loader, train_counter, train_losses, train_accs):
    network.train()
    
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
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
                (batch_idx*5) + ((epoch-1)*len(train_loader.dataset)))


# test the network as described in the tutorial
def test(network, test_loader, test_losses, test_accs):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
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
def train(train_dataloader, test_dataloader, model_path, epochs=5, lr=0.01, device="cpu"):
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    pretrained_model = torch.load(model_path)

    network = MyNetwork()
    network.load_state_dict(pretrained_model)
    for param in network.parameters():
        param.requires_grad = False

    # Replace last layer with 3 output nodes. The activation funciton (log_softmax) remains the same
    del network.fc2
    network.fc2 = nn.Linear(50, 3)

    # network = network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.5)

    train_losses = []
    train_accs = []
    train_counter = []
    test_losses = []
    test_accs = []
    test_counter = [(i+1) * len(train_dataloader.dataset) for i in range(epochs)]    

    for t in range(epochs):
        # print(f"===== Epoch: {t+1} =====")
        train_single_epoch(t+1, network, optimizer, train_dataloader, train_counter, train_losses, train_accs)
        test(network, test_dataloader, test_losses, test_accs)
        print(f"======================")
    print("Complete!")

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')

    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('# training examples seen')
    plt.ylabel('Negative log likelihood loss')
    fig.savefig("./train_test_loss_greek.png")

    fig = plt.figure()
    plt.plot(train_counter, train_accs, color='blue')
    plt.scatter(test_counter, test_accs, color='red')

    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('# training examples seen')
    plt.ylabel('Accuracy')
    fig.savefig("./train_test_acc_greek.png")


# Read greek letters from 2 separate directories
def prepare_training_data(train_path: str, test_path:str, batch_size=5) -> Tuple[DataLoader, DataLoader]:
    greek_train = DataLoader(
            torchvision.datasets.ImageFolder(train_path,
                                             transform=torchvision.transforms.Compose([
                                                 ToTensor(),
                                                 GreekTransform(),
                                                 Normalize(
                                                     (0.1307,),
                                                     (0.3081,)
                                                     )]))
                                                 ,batch_size=batch_size, shuffle=True)

    greek_test = DataLoader(
            torchvision.datasets.ImageFolder(test_path,
                                             transform=torchvision.transforms.Compose([
                                                 ToTensor(),
                                                 GreekTransform(),
                                                 Normalize(
                                                     (0.1307,),
                                                     (0.3081,)
                                                     )]))
                                                 ,batch_size=batch_size, shuffle=True)

    print("Test data info:")
    for X, y in greek_test:
        print(f"Shape of X [N, C, H, W] {X.shape}, dtype: {X.dtype}")
        print(f"Shape of y [N] {y.shape}, dtype: {y.dtype}")
        break
    return greek_train, greek_test

# main function, accepts a model pth file, greek_train_dir, greek_test_dir
def main(argv):
    if (len(argv) < 4):
        print("Usage: python q1.py <model_path> <train_dir> <test_dir>")
        sys.exit(-1)
    greek_train, greek_test = prepare_training_data(argv[2], argv[3])
    train(greek_train, greek_test, argv[1], epochs=8)
    return 0

if __name__ == "__main__":
    main(sys.argv)
