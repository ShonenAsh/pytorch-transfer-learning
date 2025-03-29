# Authors: Ashish Magadum & Nicholas Payson
# CS5330 PRCV Spring 2025

# Repo: ShonenAsh/pytorch-transfer-learning
Transfer learning experiments using PyTorch

### Number of late days used: 1

# Initial Setup

1. Please ensure that Python 3.12+ is installed
2. Create a virtual env and activate it
3. `pip install -r requirements.txt`
4. Run the below commands

# Model files, data

[link](https://drive.google.com/drive/folders/1kL3WRZstqGbMA0qiV7AeoOMG4rSkh3K-?usp=sharing)


# Files and run command:

- q1.py: Train MNIST digit classifier and save to model_dir
    `python q1.py <model_dir> <mnist_path>`

- test_handwritten.py: Run test on a dir containing handwritten samples (28x28)
    `python test_handwritten.py <model_pth_file> <handwritten_img_dir>`

- test_mnist.py: Run test on first 10 images of MNIST test dataset
    `python test_mnist.py <model_pth_file> <mnist_path>`

- q2.py: Analyze the first conv layer
    `python q2.py <model_pth_file> <mnist_dir>`

- q3.py: Transfer learning on greek letters
    `python q2.py <model_pth_file> <greek_train_dir> <greek_test_dir>`

- q4.py: experimental analysis and save to model_dir
    `python q4.py <model_dir> <mnist_path>`

- extension.py
    `python extension.py <model_pth_file>`

# Other files

- model.py: Contains the model class definition

- requirements.txt: pip dependency file
