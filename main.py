"""
Nicholas Payson & Ashish A. Magadum
CS5330 Computer Vision
Spring 2025
"""

# import statements
import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# hyperparams
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# class definitions

# My Neural Network for Task One
class MyNet(nn.Module):

    # set up the network according to prof's instructions
    def __init__(self):
        super(MyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 10,  kernel_size=5),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(p=0.1),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.LogSoftmax()
        )

    # Do a forward pass in the network
    def forward(self, x):
        return self.network(x)

#import and return a network from given filepath
def import_network_from_file(filepath):
  network = MyNet()
  state_dict = torch.load(filepath)
  network.load_state_dict(state_dict)
  return network

# Convert to Grayscale
class GrayscaleTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        return torchvision.transforms.functional.invert(x)



# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('/files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # show first six numbers in MNIST
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    network = MyNet()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, optimizer, train_loader, train_counter, train_losses)
        test(network, test_loader, test_losses)

    # Show training loss graph
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig.show()

    return

# train the network as described in the tutorial
def train(epoch, network, optimizer, train_loader, train_counter, train_losses):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'C:/Users/nickp/Downloads/results/model.pth')
      torch.save(optimizer.state_dict(), 'C:/Users/nickp/Downloads/results/optimizer.pth')

# test the network as described in the tutorial
def test(network, test_loader, test_losses):
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
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


# call main when run from command line, passing in argv (which are ignored here)
if __name__ == "__main__":
    main(sys.argv)