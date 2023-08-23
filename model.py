# -*- coding: utf-8 -*-


"""
The file for the definition of classifier model.
    CNNClassifier - Class for the EfficientNet Classifier Model.
    SWINClassifier - Class for the SWIN Transformer Classifier Model.
"""


# Library Imports
import timm
import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2023, Calibration Where it Matters"
__credits__   = ["Jacob Carse"]
__license__   = "MIT"
__version__   = "1.0.0"
__maintainer  = ["Jacob Carse"]
__email__     = ["j.carse@dundee.ac.uk"]
__status__    = "Development"


class CNNClassifier(nn.Module):
    """
    Class for the Classifier model that uses an EfficientNet encoder.
        init - Initialiser for the model.
        forward - Performs forward propagation.
    """

    def __init__(self, b: int = 0, pretrained: bool = True) -> None:
        """
        Initialiser for the model that initialises the model's layers.
        :param b: The compound coefficient of the EfficientNet model to be loaded.
        :param pretrained: If the pretrained weights should be loaded.
        """

        # Calls the super for the nn.Module.
        super(CNNClassifier, self).__init__()

        # Loads the EfficientNet encoder.
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(b)}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{str(b)}")

        # Defines the Pooling layer for the Encoder outputs.
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Defines the hidden Fully Connected Layer.
        self.hidden = nn.Linear(self.encoder._fc.in_features, 512)

        # Defines the output Fully Connected Layer.
        self.classifier = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation with the Classifier.
        :param x: Input image batch.
        :return: PyTorch Tensor of logits.
        """

        # Performs forward propagation with the encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Performs forward propagation with the hidden layer.
        x = F.silu(self.hidden(x))

        # Gets the output logits from the output layer.
        return self.classifier(x)


class SWINClassifier(nn.Module):
    """
    Class for the Classifier model that uses an SWIN Transformer encoder.
        init - Initialiser for the model.
        forward - Performs forward propagation.
    """

    def __init__(self, pretrained: bool = True) -> None:
        """
        Initialiser for the model that initialises the model's layers.
        :param class_num: The number of classes the model will be predicting.
        :param pretrained: If the pretrained weights should be loaded.
        """

        # Calls the super for the nn.Module.
        super(SWINClassifier, self).__init__()

        # Loads the SWIN transformer encoder.
        self.encoder = timm.create_model("swin_base_patch4_window7_224_in22k",
                                         pretrained=pretrained, num_classes=1)

        # Defines the hidden Fully Connected Layer.
        self.hidden = nn.Linear(1024, 512)

        # Defines the output Fully Connected Layer.
        self.classifier = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation with the Classifier.
        :param x: Input image batch.
        :return: PyTorch Tensor of logits.
        """

        # Performs forward propagation with the encoder.
        x = self.encoder.forward_features(x)
        x = x.mean(dim=1)

        # Performs forward propagation with the hidden layer.
        x = F.gelu(self.hidden(x))

        # Gets the output logits from the output layer.
        return self.classifier(x)
