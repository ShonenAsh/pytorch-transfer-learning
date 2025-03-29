# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# This file contains the model architecture

import torch.nn.functional as F
import torch.nn as nn
import torch

# Model definition
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


    # This applies activation functions on the layers defined above in the init function
    # computes a forward pass for the network
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


#import and return a network from given filepath
def import_network_from_file(filepath):
    network = MyNetwork()
    state_dict = torch.load(filepath)
    network.load_state_dict(state_dict)
    return network
