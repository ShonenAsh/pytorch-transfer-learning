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

    # set up the network according params
    # each param is a list like this with lists inside, one internal list for each layer:
    # [[list a ], [list b]]
    # each list looks like:
    # [param1, param2, param3 ...] and has the values for the params in order for the corresponding type of layer
    def __init__(self, conv_params, dropout_params, linear_params):
        super(MyNet, self).__init__()

        layers = []
        for conv in conv_params:
            layers.append(nn.Conv2d(conv[0], conv[1], kernel_size=conv[2]))
            layers.append(nn.MaxPool2d(2, 2, 0))
            layers.append(nn.ReLU())

        for dropout in dropout_params:
            layers.append(nn.Dropout2d(p=dropout[0]))
            layers.append(nn.ReLU())

        layers.append(nn.Flatten())

        for linear in linear_params:
            layers.append(nn.Linear(linear[0], linear[1]))
            layers.append(nn.ReLU())

        layers.append(nn.LogSoftmax())
        self.network = nn.Sequential(*layers)
        """nn.Conv2d(1, 10,  kernel_size=5),
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
            nn.LogSoftmax()"""

    # Do a forward pass in the network
    def forward(self, x):
        return self.network(x)

def get_score_for_params(conv, drop, lin):
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

    network = MyNet(conv, drop, lin)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    correctness = test(network, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch, network, optimizer, train_loader, train_counter, train_losses)
        correctness = test(network, test_loader, test_losses)

    return correctness


# main function (yes, it needs a comment too)
def main(argv):
    # Search Strategy:
    # Alternate running through all possibilities for conv, drop, and linear layers.
    # Get the best settings for one type while holding the other two constant, then switch which two are constant.
    # Iterate through optimizing each subsection of the network three times each
    # to find the settings that work best together
    # each network gets three epochs and 60000 samples per epoch to train before being tested

    #  set up all permutations of params to test

    # All convs have to get to 20 layers after two, only testing two conv layers
    # Linear layers all start with 320 which limits the combinations of things we can do here
    all_convs = [
    [[1, 10, 5], [10, 20, 5]],
    [[1, 5, 5], [5, 20, 5]],
    [[1, 15, 5], [15, 20, 5]],
    [[1, 7, 5], [7,20,5]],
    [[1, 12, 5], [12, 20, 5]],
    [[1, 18, 5], [18, 20, 5]],
    [[1, 3, 5], [3, 20, 5]]]

    all_drops = [
        [[0.05], [0.05]],
        [[0.15], [0.25]],
        [[0.05], [0.25]],
        [[0.15], [0.25]],
        [[0.25], [0.25]],
        [[0.25], [0.45]],
        [[0.35], [0.45]],
        [[0.45], [0.45]],
        [[0.1]],
        [[0.15]],
        [[0.2]],
        [[0.25]],
        [[0.3]],
        [[0.35]],
        [[0.4]],
        [[0.45]],
        [[0.5]]]

    all_linears = [
        [[320, 50], [50, 10]],
        [[320, 100], [100, 10]],
        [[320, 150], [150, 10]],
        [[320, 200], [200, 10]],
        [[320, 160], [160, 10]],
        [[320, 40], [40, 10]],
        [[320, 80], [80, 10]],
        [[320, 20], [20, 10]],
        [[320, 250], [250, 150], [150, 10]],
        [[320, 200], [200, 50], [50, 10]],
        [[320, 150], [150, 100], [100, 10]],
        [[320, 150], [150, 50], [50, 10]],
        [[320, 100], [100, 50], [50, 10]],
        [[320, 200], [200, 150], [150, 10]],
        [[320, 280], [280, 200], [200, 10]],
        [[320, 80], [80, 40], [40, 10]],
        [[320, 10]]]

    best_conv_index = 0
    best_drop_index = 0
    best_linear_index = 0
    final_best_score = 0
    for main_iterations in range(3):

        current_best_score = 0
        current_best_index = 0
        print(f"Running convs {main_iterations + 1}")
        for conv_index in range(len(all_convs)):

            score = get_score_for_params(all_convs[conv_index], all_drops[best_drop_index], all_linears[best_linear_index])

            if score > current_best_score:
                current_best_score = score
                current_best_index = conv_index

        if current_best_score > final_best_score:
            final_best_score = current_best_score
            best_conv_index = current_best_index

        current_best_score = 0
        current_best_index = 0
        print(f"Running drops {main_iterations + 1}")
        for drop_index in range(len(all_drops)):

            score = get_score_for_params(all_convs[best_conv_index], all_drops[drop_index],
                                         all_linears[best_linear_index])

            if score > current_best_score:
                current_best_score = score
                current_best_index = drop_index

        if current_best_score > final_best_score:
            final_best_score = current_best_score
            best_drop_index = current_best_index

        current_best_score = 0
        current_best_index = 0
        print(f"Running linears {main_iterations + 1}")
        for lin_index in range(len(all_linears)):

            score = get_score_for_params(all_convs[best_conv_index], all_drops[best_drop_index],
                                         all_linears[lin_index])

            if score > current_best_score:
                current_best_score = score
                current_best_index = lin_index

        if current_best_score > final_best_score:
            final_best_score = current_best_score
            best_linear_index = current_best_index

    best_net = MyNet(all_convs[best_conv_index], all_drops[best_drop_index], all_linears[best_linear_index])
    print("Optimized Network: ")
    print(best_net)
    print(f"Optimized Network Test Accuracy: {final_best_score}")

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
      """print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))"""
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

  return 100. * correct / len(test_loader.dataset)


# call main when run from command line, passing in argv (which are ignored here)
if __name__ == "__main__":
    main(sys.argv)