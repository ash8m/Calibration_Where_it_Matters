#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""
Executable for experiments for Calibration Where it Matters.
"""


# Built-in/Generic Imports
import warnings

# Own Modules Imports
# TODO from test import test_classifier
from train import train_classifier
from utils import log, set_random_seed
from config import load_configurations, print_arguments


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


if __name__ == "__main__":
    # Loads the arguments from configurations file and command line.
    description = "Experiments on Calibration Where it Matters"
    arguments = load_configurations(description)

    # Displays the loaded arguments.
    log(arguments, "Loaded Arguments:")
    print_arguments(arguments)

    # Removes warnings from output.
    if not arguments.warning:
        warnings.filterwarnings("ignore")

    # Sets random seed.
    if arguments.seed != -1:
        set_random_seed(arguments.seed)
        log(arguments, f"Set Random Seed to {arguments.seed}")

    # Trains a CNN model.
    if arguments.task.lower() == "train":
        train_classifier(arguments)

    # Tests a CNN model.
    elif arguments.task.lower() == "test":
        pass
        # TODO test_cnn(arguments)

    else:
        log(arguments, "Enter a valid task. \"train\" or \"test\"")
