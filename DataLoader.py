from ntpath import join
import time
import os
import copy
from collections import defaultdict

# deep learning/vision libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt

#Ovaj kod koristi dosta koda iz CNN WS-a koji mozete naci na: https://github.com/Petlja/PSIML/tree/master/workshops/CNNs


def get_Dataset(folder_name = "new_data", bs = 10, shuffle = True):

    img_path = os.path.join(os.getcwd(),folder_name)
    train_val = ['train','val']

    transformations = torchvision.transforms.Compose([transforms.Grayscale(num_output_channels = 1),transforms.ToTensor()])


    img_datasets = {x:datasets.ImageFolder(os.path.join(img_path,x),transformations) for x in train_val}

    dataloaders = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size = bs, shuffle = shuffle) for x in train_val}

    return dataloaders


def img_batch_show_test(img_dataset):
    dataset_sizes = {x: len(img_dataset[x]) for x in ['train', 'val']}
    print(f'Training dataset size: {dataset_sizes["train"]} images, Validation dataset size: {dataset_sizes["val"]} images')



    img_batch, classes = next(iter(img_dataset['train'])) # Get a batch of training data
    print(f'Shape of batch of images: {img_batch.shape}')

    grid =torchvision.utils.make_grid(img_batch)

    plt.imshow(grid.numpy().transpose((1,2,0)))

    plt.show()

