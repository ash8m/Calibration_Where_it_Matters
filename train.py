# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to train a CNN model.
    train_classifier - Function used to train a Convolutional Neural Network.
"""


# Built-in/Generic Imports
import os
from time import time
from argparse import Namespace

# Library Imports
import torch
import lightning as L
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

# Own Modules
from utils import log
from dataset import get_datasets
from model import CNNClassifier, SWINClassifier


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


def train_classifier(arguments: Namespace) -> None:
    """
    Function for training the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    """

    # Sets up a Fabric Lightning accelerator.
    fabric = L.Fabric(accelerator="auto", devices="auto", strategy="auto",
                      precision="16-mixed" if arguments.mixed_precision else "32-true")

    # Initialises the Lighting accelerator.
    fabric.launch()

    # Sets PyTorch to detect errors in Autograd, useful for debugging but slows down performance.
    if arguments.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Loads the training and validation data.
    train_data, val_data, _ = get_datasets(arguments)

    # Creates the training data loader using the dataset object.
    train_data_loader = DataLoader(train_data, batch_size=arguments.batch_size,
                                   shuffle=True, num_workers=arguments.data_workers,
                                   pin_memory=False, drop_last=False)

    # Creates the validation data loader using the dataset object
    valid_data_loader = DataLoader(val_data, batch_size=arguments.batch_size * 2,
                                   shuffle=False, num_workers=arguments.data_workers,
                                   pin_memory=False, drop_last=False)

    # Sets up the data loaders with Fabric, so they load of the correct devices with the correct precision.
    train_data_loader, valid_data_loader = fabric.setup_dataloaders(train_data_loader, valid_data_loader)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    if arguments.swin_model:
        # Loads the SWIN Transformer model.
        classifier = SWINClassifier()
    else:
        # Loads the EfficientNet CNN model.
        classifier = CNNClassifier(arguments.efficient_net)

    # Sets the classifier to training mode.
    classifier.train()

    # Initialises the optimiser used to optimise the parameters of the model.
    optimiser = SGD(params=classifier.parameters(), lr=arguments.minimum_lr)

    # Initialises the learning rate scheduler to adjust the learning rate during training.
    step_size = (len(train_data_loader) // arguments.batch_size) * 2
    scheduler = lr_scheduler.CyclicLR(optimiser, base_lr=arguments.minimum_lr, max_lr=arguments.maximum_lr,
                                      step_size_up=step_size, mode="triangular")

    # Sets up the model and optimiser with Fabric, so they load of the correct devices with the correct precision.
    classifier, optimiser = fabric.setup(classifier, optimiser)

    log(arguments, "Model Initialised")
