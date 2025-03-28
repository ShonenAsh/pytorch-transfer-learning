# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# This file trains and tests a Convolutional neural network on the MNIST handwritten digits dataset

import torch
from torch import nn as nn
from torch._prims_common import Tensor
from torch.cuda import is_available
import torch.nn.functional as F
from torch.utils.data import DataLoader 
from torchvision import datasets
import torchvision
from torchvision.transforms import Normalize, ToTensor
from typing import Tuple
import sys
from datetime import datetime
import matplotlib.pyplot as plt

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        # x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

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
    fig.savefig("train_test_loss_greek.png")

    fig = plt.figure()
    plt.plot(train_counter, train_accs, color='blue')
    plt.scatter(test_counter, test_accs, color='red')

    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper right')
    plt.xlabel('# training examples seen')
    plt.ylabel('Accuracy')
    fig.savefig("train_test_acc_greek.png")



def prepare_training_data(path: str, batch_size=5) -> Tuple[DataLoader, DataLoader]:
    greek_train = DataLoader(
            torchvision.datasets.ImageFolder("./data/greek_train",
                                             transform=torchvision.transforms.Compose([
                                                 ToTensor(),
                                                 GreekTransform(),
                                                 Normalize(
                                                     (0.1307,),
                                                     (0.3081,)
                                                     )]))
                                                 ,batch_size=batch_size, shuffle=True)

    greek_test = DataLoader(
            torchvision.datasets.ImageFolder("./data/greek_test",
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

# main function (yes, it needs a comment too)
def main(argv):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"
    greek_train, greek_test = prepare_training_data("./data/")
    train(greek_train, greek_test, "./model/model_2025-03-27T15:05:52.pth", epochs=8, device=DEVICE)
    return 0

if __name__ == "__main__":
    main(sys.argv)
