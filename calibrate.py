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

    def train(self, logits: np.ndarray, labels: np.ndarray) -> None:
        """
        With the provided logits and labels will learn a temperature parameter to calibrate the model.
        :param logits:
        :param labels:
        """

        def negative_log_loss(temperature: list, *args) -> float:
            """
            Get the negative log loss using the logits, labels and provided temperature parameter.
            :param temperature:
            :param args:
            :return:
            """

            logits, labels = args
            epsilon = 1e-7
            logits = np.clip(logits, epsilon, 1 - epsilon)
            loss = -np.mean(labels * np.log(logits) + (1 - labels) * np.log(1 - logits))
            return loss

        self.temperature = optimize.minimize(negative_log_loss, 1.0, args=(logits, labels), method="L-BFGS-B",
                                             bounds=((0.05, 5.0),), tol=1e-15, options={"disp": self.verbose}).x[0]

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
