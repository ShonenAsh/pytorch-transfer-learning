# import statements
import sys
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import MyNet, batch_size_test, import_network_from_file

# import the network from the given filepath and return it



#main function
def main():
    network = import_network_from_file("C:/Users/nickp/Downloads/results/model.pth")
    network.eval()

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

    with torch.no_grad():
        output = network(example_data)

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        print()
        print("Chosen Item: ")
        print(output.data.max(1, keepdim=True)[1][i].item())
        list_tensor = output.data[i].tolist()
        print("Output Values: ")
        for number, item in zip(range(10), list_tensor):
            print(f"{number} : {item:.2f}")

        plt.xticks([])
        plt.yticks([])
    fig.show()
    return


if __name__ == "__main__":
    main()
