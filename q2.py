# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025
# print out the filters from the first layer, show them as heatmaps
# also show their effects on first image with the filepath given in argv
#
# The main function accepts a model_pth file and mnist_data_dir


import sys

import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import Normalize, ToTensor

from model import import_network_from_file


# main function, accepts a model_pth file and mnist_data_dir
def main(argv):
    if (len(argv) < 3):
        print("Usage: python q2.py <model_pth_file> <mnist_dir>")
        sys.exit(-1)

    network = import_network_from_file(argv[1])
    network.eval()
    print(network)
    mnist_data = torchvision.datasets.MNIST(argv[2],
                                            train=True,
                                            download=True,
                                             transform=torchvision.transforms.Compose([
                                                 ToTensor(),
                                                 Normalize(
                                                     (0.1307,),
                                                     (0.3081,))
                                                 ]))
    first_img, first_label = mnist_data[0]
    first_img_np = first_img.squeeze().numpy()
    
    fig, axes = plt.subplots(3, 4, figsize=(9, 12))
    axes = axes.flatten() 
    # First layer (works regardless of named parameters)
    for name, param in network.named_parameters():
        print(f"layer: {name}")
        for i in range(12):
            # To hide the remaining plots
            if i < 10:
                curr_filter = param.data[i][0]
                axes[i].imshow(curr_filter, cmap="hot", interpolation="nearest")
                axes[i].set_title(f"Filter {i+1} shape: {curr_filter.shape[0]}x{curr_filter.shape[1]}")
            axes[i].axis("off")

        break # only first param.
    
    plt.suptitle(f"Filters of the first conv layer")
    fig.savefig(f"./model_filters.png")



    
    fig, axes = plt.subplots(5, 4, figsize=(12, 12))
    
    # First layer (works regardless of named parameters)
    for name, param in network.named_parameters():
        print(f"layer: {name}")
        for i in range(5):
            for j in range(2):
                filter_id = i*2+j
                curr_filter = param.data[filter_id][0]
                print(f"Filter {filter_id+1}: {curr_filter.shape}")

                axes[i, j*2].imshow(curr_filter, cmap="hot", interpolation="nearest")
                axes[i, j*2].set_title(f"Filter {filter_id + 1}")
                axes[i, j*2].axis("off")

                filtered_img = cv2.filter2D(first_img_np, -1, curr_filter.squeeze().numpy())

                axes[i, j*2+1].imshow(filtered_img, cmap="hot", interpolation="nearest")
                axes[i, j*2+1].set_title(f"Filtered image")
                axes[i, j*2+1].axis("off")
        break # only first param.
    
    plt.suptitle(f"Affects of learned filters on MNIST digit: {first_label}")
    fig.savefig(f"./model_filter_analysis.png")
    return

# Main function, accepts a model_path and mnist_data_dir
if __name__ == "__main__":
    main(sys.argv)
