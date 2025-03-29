# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# Test custom handwritten digits (28x28)

# import statements
import sys
import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale, invert
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import import_network_from_file

# Grayscale and inversion of intensities
class GrayscaleTransform:

    def __init__(self):
        pass

    def __call__(self, x):
        x = rgb_to_grayscale( x )
        return invert(x)

# main function, accepts a model_pth_file and dir containing handwritten samples (28x28)
def main(argv):
    if (len(argv) < 3):
        print("Usage: python test_handwritten.py <model_pth_file> <mnist_path>")
        sys.exit(-1)
        
    network = import_network_from_file(argv[1])
    network.eval()

    # Load custom dataset
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(argv[2],
                                         transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                   GrayscaleTransform(),
                                                                                   torchvision.transforms.Normalize(
                                                                                       (0.1307,), (0.3081,))])),
        batch_size=10,
        shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        output = network(example_data)

    # Plot predictions
    fig = plt.figure()
    for i in range(10):
        plt.subplot(5, 2, i + 1)
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
    fig.savefig("./test_handwritten_MNIST.png")

    return

if __name__ == "__main__":
    main(sys.argv)
