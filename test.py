# -*- coding: utf-8 -*-


"""
The file contains implementations of the functions used to test a classifier model.
    test_classifier - Function for testing a classifier model with optional calibration.
"""


# Built-in/Generic Imports
import os
from argparse import Namespace

# Library Imports
import tqdm
import torch
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

# Own Modules
from utils import log
from calibrate import calibrate_model
from dataset import Dataset, get_datasets
from model import CNNClassifier, ResNetClassifier


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


def test_classifier(arguments: Namespace):
    """
    Function for training a classifier model.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    """

    # Sets up a Fabric Lightning accelerator.
    fabric = L.Fabric(accelerator="auto", devices="auto", strategy="auto",
                      precision="16-mixed" if arguments.mixed_precision else "32-true")

    # Initialises the Lighting accelerator.
    fabric.launch()

    # Loads the testing and validation data.
    _, val_data, test_data = get_datasets(arguments)

    # Creates the validation data loader using the dataset object.
    val_data_loader = DataLoader(val_data, batch_size=arguments.batch_size * 2, shuffle=False,
                                 num_workers=arguments.data_workers, pin_memory=False, drop_last=False)

    # Creates the testing data loader using the dataset object.
    test_data_loader = DataLoader(test_data, batch_size=arguments.batch_size * 2, shuffle=False,
                                  num_workers=arguments.data_workers, pin_memory=False, drop_last=False)

    # Sets up the data loaders with Fabric, so they load of the correct devices with the correct precision.
    val_data_loader, test_data_loader = fabric.setup_dataloaders(val_data_loader, test_data_loader)

    log(arguments, "Loaded Datasets")

    # Initialises the classifier model.
    if arguments.swin_model:
        # Loads the SWIN Transformer model.
        classifier = ResNetClassifier(arguments.resnet_layers)
    else:
        # Loads the EfficientNet CNN model.
        classifier = CNNClassifier(arguments.efficient_net)

    # Loads the trained model.
    model_name = f"{arguments.experiment}_{arguments.dataset}.pt"
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, model_name)))

    # Sets the classifier to testing mode.
    classifier.eval()

    # Sets up the model with Fabric, so they load of the correct devices with the correct precision.
    classifier = fabric.setup(classifier)

    log(arguments, "Model Loaded")

    if arguments.calibration_method in ["temperature"]:
        batch_count = 0
        logit_list = torch.tensor([], device=classifier.device)
        label_list = torch.tensor([], device=classifier.device, dtype=torch.int64)

        with torch.no_grad():
            for images, labels in val_data_loader:
                logits = classifier(images)
                logits = logits.view(logits.shape[0])

                if arguments.boundary_calibration:
                    indices = ((logits <= 0).nonzero(as_tuple=True)[0])
                    label_list = torch.cat((label_list, labels[indices]))
                    logit_list = torch.cat((logit_list, logits[indices]))
                else:
                    label_list = torch.cat((label_list, labels))
                    logit_list = torch.cat((logit_list, logits))

                batch_count += 1
                if batch_count == arguments.batches_per_epoch:
                    break

        calibrator = calibrate_model(arguments, logit_list, label_list)

        log(arguments, "Calibrated Model")

    # Defines the batch count and initialises the list of predictions and labels.
    batch_count, prediction_list, label_list = 0, [], []

    # Loops through the testing data batches with no gradient calculations.
    with torch.no_grad():
        with tqdm.tqdm(test_data_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description("Testing")

                # Appends the labels to the array of labels.
                label_list += list(labels.cpu().numpy())

                # Performs forward propagation with the model to get the output logits.
                logits = classifier(images).view(images.shape[0])

                # Calibrate predictions using the calibrator or just use sigmoid.
                if arguments.calibration_method in ["temperature"]:
                    predictions = calibrator(logits)
                else:
                    predictions = torch.sigmoid(logits)

                # Moves the predictions to the CPU.
                predictions = predictions.cpu().numpy()

                # Gets the predictive probabilities and appends them to the array of predictions.
                prediction_list += list(predictions)

                # Adds to the current batch count.
                batch_count += 1

                # If the number of batches have been reached end testing.
                if batch_count == arguments.batches_per_epoch:
                    break

    # Creates the output directory for the output files.
    os.makedirs(arguments.output_dir, exist_ok=True)

    # Creates the DataFrame from the labels and output predictions.
    data_frame = pd.DataFrame(list(zip(label_list, prediction_list)), columns=["label", "prediction"])

    # Outputs the output DataFrame to a csv file.
    boundary = f"_{arguments.boundary_calibration}" if arguments.calibration_method == "temperature" else ''
    output_name = f"{arguments.experiment}_{arguments.dataset}_{arguments.calibration_method}{boundary}.csv"
    data_frame.to_csv(os.path.join(arguments.output_dir, output_name), index=False)
