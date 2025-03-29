# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# Run test on first 10 images of MNIST test dataset

# import statements
import sys
import torch
import torchvision
import matplotlib.pyplot as plt

from model import import_network_from_file

# main function, accepts a model_pth_file and dir containing handwritten samples (28x28)
def main(argv):
    if (len(argv) < 3):
        print("Usage: python test_mnist.py <model_pth_file> <mnist_path>")
        sys.exit(-1)
    network = import_network_from_file(argv[1])
    network.eval()

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(argv[2], train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=1000, shuffle=False)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    with torch.no_grad():
        output = network(example_data)

    # Save first 6 examples (1A task)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    fig.savefig("./first_6_examples.png")


    # Test on MNIST test data
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
    fig.savefig("./first_10_examples_test.png")
    return


if __name__ == "__main__":
    main(sys.argv)
