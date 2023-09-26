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
import tqdm
import torch
import lightning as L
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

# Own Modules
from utils import log
from dataset import get_datasets
from model import CNNClassifier, ResNetClassifier


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
    if arguments.resnet_model:
        # Loads the ResNet CNN model.
        classifier = ResNetClassifier(arguments.binary, arguments.resnet_layers)
    else:
        # Loads the EfficientNet CNN model.
        classifier = CNNClassifier(arguments.binary, arguments.efficient_net)

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

    # Declares the main logging variables for training.
    start_time = time()
    best_loss, best_epoch = 1e10, 0

    # The beginning of the main training loop.
    for epoch in range(1, arguments.epochs + 1):
        # Declares the logging variables for the training epoch.
        epoch_acc, epoch_loss, epoch_batches = 0., 0., 0

        # Loops through the training data batches.
        with tqdm.tqdm(train_data_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                # Resets the gradients in the model.
                optimiser.zero_grad()

                # Performs forward propagation using the classifier model.
                predictions = classifier(images)
                if arguments.binary:
                    predictions = predictions.view(images.shape[0])

                # Calculates the binary cross entropy loss.
                if arguments.binary:
                    loss = F.binary_cross_entropy_with_logits(predictions, labels.float())
                else:
                    loss = F.cross_entropy(predictions, labels)

                # Performs backward propagation with the loss.
                fabric.backward(loss)

                # Updates the parameters of the model using the optimiser.
                optimiser.step()

                # Updates the learning rate scheduler.
                scheduler.step()

                # Calculates the accuracy of the batch.
                if arguments.binary:
                    batch_accuracy = 1 - ((torch.round(predictions) == labels).sum().double() / labels.shape[0])
                else:
                    batch_accuracy = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                # Adds the number of batches, losses and accuracy to the epoch sum.
                epoch_batches += 1
                epoch_loss += loss.item()
                epoch_acc += batch_accuracy.item()

                # Logs the loss and accuracy of the batch
                tepoch.set_postfix(loss=epoch_loss / epoch_batches, accuracy=epoch_acc / epoch_batches)

                # If the number of batches have been reached end training.
                if epoch_batches == arguments.batches_per_epoch:
                    break

        # Declares the logging variables for the validation epoch.
        val_acc, val_loss, val_batches = 0., 0., 0

        # Loops through the validation data batches with no gradient calculations.
        with (torch.no_grad()):
            with tqdm.tqdm(valid_data_loader, unit="batch") as tepoch:
                for images, labels in tepoch:
                    tepoch.set_description(f"Validation Epoch {epoch}")

                    # Performs forward propagation using the CNN model.
                    predictions = classifier(images)
                    if arguments.binary:
                        predictions = predictions.view(images.shape[0])

                    # Calculates the cross entropy loss.
                    if arguments.binary:
                        loss = F.binary_cross_entropy_with_logits(predictions, labels.float())
                    else:
                        loss = F.cross_entropy(predictions, labels)

                    # Calculates the accuracy of the batch.
                    if arguments.binary:
                        batch_accuracy = 1 - ((torch.round(predictions) == labels).sum().double() / labels.shape[0])
                    else:
                        batch_accuracy = (predictions.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                    # Adds the number of batches, losses and accuracy to the epoch sum.
                    val_batches += 1
                    val_loss += loss.item()
                    val_acc += batch_accuracy.item()

                    # If the number of batches have been reached end validation.
                    if val_batches == arguments.batches_per_epoch:
                        break

        # Logs the details of the training epoch.
        log(arguments, "Epoch: {}\tTime: {:.1f}s\tTraining Loss: {:.6f}\tTraining Accuracy: {:.6f}\n"
                       "Validation Loss: {:.6f}\tValidation Accuracy: {:.6f}\n".format(
                       epoch, time() - start_time, epoch_loss / epoch_batches, epoch_acc / epoch_batches,
                       val_loss / val_batches, val_acc / val_batches))

        # If the current epoch has the best validation loss then save the model.
        if val_loss / val_batches < best_loss:
            best_loss = val_loss / val_batches
            best_epoch = epoch

            # Checks if the save directory exists and if not create it.
            os.makedirs(arguments.model_dir, exist_ok=True)

            # Saves the model to the save directory.
            model_name = f"{arguments.experiment}_{arguments.dataset}.pt"
            torch.save(classifier.state_dict(), os.path.join(arguments.model_dir, model_name))

    # Logs final training information.
    log(arguments, f"Training finished with best loss of {round(best_loss, 4)} at epoch {best_epoch} in "
                   f"{int(time() - start_time)}s.")
