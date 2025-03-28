# import statements
import sys

import matplotlib
import numpy as np
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from main import GrayscaleTransform, import_network_from_file
#import cv2

def main(argv):
    # print out the filters from the first layer, show them as heatmaps
    # also show their effects on first image with the filepath given in argv
    network = import_network_from_file("C:/Users/nickp/Downloads/results/model.pth")
    network.eval()

    #mnist_data = pd.read_csv("C:/Users/nickp/Downloads/mnist_train.csv/mnist_train.csv")
    #first_image_pixels = mnist_data.iloc[0, 1:].values
    #first_image = first_image_pixels.reshape(28, 28).astype(np.uint8)  # just in case not already grayscale

    param_num = 0
    for name, param in network.named_parameters():

        if param_num == 0:  # first layer
            for i in range(10):
                the_filter = param.data[i][0]
                plt.title(f"Filter {i + 1}")
                plt.imshow(the_filter, cmap="hot", interpolation="nearest")
                plt.show()
                #img = cv2.filter2D(src=first_image, ddepth=-1, kernel=the_filter)
                #cv2.imshow('Filtered Image', img)

        param_num += 1



    #For printing entire network structure
    #print network
    #print(network)

    return

if __name__ == "__main__":
    main(sys.argv)
