# -*- coding: utf-8 -*-


"""
The file contains the following utility functions for the application:
    log - Function to print and/or log messages to the console or logging file.
    str_to_bool - Function to convert an input string to a boolean value.
    set_random_seed - Function used to set the random seed for all libraries used to generate random numbers.
"""


# Built-in/Generic Imports
import os
from argparse import ArgumentTypeError, Namespace

# Library Imports
from lightning import seed_everything


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


def log(arguments: Namespace, message: str) -> None:
    """
    Logging function that will both print and log an input message.
    :param arguments: ArgumentParser object containing "log_dir" and "experiment".
    :param message: String containing the message to be printed and/or logged.
    """

    # Prints the message to the console if verbose is set to True.
    if arguments.verbose:
        print(message)

    if arguments.log_dir != '':
        # Creates the directory for the log file.
        os.makedirs(arguments.log_dir, exist_ok=True)

        # Logs the message to the log file.
        print(message,
              file=open(os.path.join(arguments.log_dir, f"{arguments.experiment}_{arguments.dataset}_log.txt"), 'a'))


def str_to_bool(argument: str) -> bool or ArgumentTypeError:
    """
    Function to convert a string to a boolean value.
    :param argument: String to be converted.
    :return: Boolean value.
    """

    # Checks if the argument is already a Boolean value.
    if isinstance(argument, bool):
        return argument

    # Returns Boolean depending on the input string.
    if argument.lower() in ["true", "t"]:
        return True
    elif argument.lower() in ["false", "f"]:
        return False

    # Returns an error if the value is not converted to a Boolean value.
    return ArgumentTypeError(f"Boolean value expected. Got \"{argument}\".")


def set_random_seed(seed: int) -> None:
    """
    Sets the random seed for all libraries that are used to generate random numbers.
    :param seed: Integer for the seed that will be used.
    """

    # Sets the seed for all libraries.
    seed_everything(seed)

