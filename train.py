# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to train a CNN model.
    train_classifier - Function used to train a Convolutional Neural Network.
"""


# Built-in/Generic Imports
import os
from argparse import Namespace

# Library Imports
import torch
from torch.utils.data import DataLoader

# Own Modules
from utils import log
from dataset import get_datasets


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


def train_classifier(arguments: Namespace, device: torch.device) -> None:
    """
    Function for training the Convolutional Neural Network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    """

    # Sets PyTorch to detect errors in Autograd, useful for debugging but slows down performance.
    if arguments.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Loads the training and validation data.
    train_data, val_data, _ = get_datasets(arguments)

    # Creates the training data loader using the dataset object.
    training_data_loader = DataLoader(train_data, batch_size=arguments.batch_size,
                                      shuffle=True, num_workers=arguments.data_workers,
                                      pin_memory=False, drop_last=False)

    # Creates the validation data loader using the dataset object
    validation_data_loader = DataLoader(val_data, batch_size=arguments.batch_size * 2,
                                        shuffle=False, num_workers=arguments.data_workers,
                                        pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")
