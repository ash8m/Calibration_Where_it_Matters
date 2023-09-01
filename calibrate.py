# -*- coding: utf-8 -*-


"""

"""


# Built-in/Generic Imports
from argparse import Namespace

# Library Imports
import torch
import numpy as np
from scipy import optimize
import torch.nn.functional as F


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


class TemperatureScaling:
    """

    """

    def __init__(self, verbose: bool = False) -> None:
        """

        :param verbose:
        """

        super(TemperatureScaling, self).__init__()
        self.verbose = verbose
        self.temperature = None

    def train(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        With the provided logits and labels will learn a temperature parameter to calibrate the model.
        :param logits: PyTorch Tensor of output model logits.
        :param labels: PyTorch Tensor of image labels.
        """

        # Sets the initial temperature parameter.
        temperature = torch.nn.Parameter(torch.ones(1))

        # Creates the temperature optimiser.
        temp_optimiser = torch.optim.LBFGS([temperature], lr=0.02, max_iter=1000, line_search_fn="strong_wolfe")

        def _eval() -> torch.Tensor:
            """
            Evaluation funtion for temperature scaling optimiser.
            :return: PyTorch Tensor for temperature scaling loss.
            """

            temp_loss = F.binary_cross_entropy_with_logits(torch.div(logits, temperature), labels)
            temp_loss.backward()
            return temp_loss

        # Uses the optimiser to optimise the eval function.
        temp_optimiser.step(_eval)

        # Sets the temperature to the optimised temperature.
        self.temperature = temperature.item()


    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Use the calibrated temperature parameter to calibrate the predicted logits.
        :param logits:
        :return:
        """

        # Calls an error if no temperature is set.
        if self.temperature is None:
            print("Error: need to first train before calling this function.")

        # Returns the softmax
        return torch.sigmoid(logits / self.temperature)


def calibrate_model(arguments: Namespace, logits: torch.Tensor, labels: torch.Tensor) -> object:
    """

    :param arguments:
    :param logits:
    :param labels:
    :return:
    """

    if arguments.calibration_method == "temperature":
        calibrator = TemperatureScaling(arguments.calibration_verbose)
        calibrator.train(logits, labels)
        return calibrator

    else:
        return None
