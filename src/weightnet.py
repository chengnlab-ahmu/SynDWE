
"""
SynDWE: Dynamic Weight Learning Network

This module implements the WeightNet architecture
used in the SynDWE dynamic weighted ensemble framework.

Components
----------
1. ResidualBlock
2. WeightNet

WeightNet learns sample-specific weights for four
base classifiers:
    - XGBoost
    - Random Forest
    - LightGBM
    - GBDT
"""

import torch
import torch.nn as nn

from src.config import (
    N_BASE_MODELS,
    WEIGHTNET_INPUT_HIDDEN,
    WEIGHTNET_RESIDUAL_UNITS,
    WEIGHTNET_INPUT_DROPOUT,
    WEIGHTNET_RESIDUAL_DROPOUT,
)


# ==================================================
# Residual Block
# ==================================================

class ResidualBlock(nn.Module):
    """
    Fully connected residual block.

    Architecture
    ------------
    Linear -> ReLU -> BatchNorm -> Dropout
              +
           Shortcut
              |
             ReLU

    Parameters
    ----------
    input_dim : int
        Dimension of input features.

    units : int
        Number of output hidden units.

    dropout_rate : float, default=0.0
        Dropout probability.
    """

    def __init__(
        self,
        input_dim,
        units,
        dropout_rate=0.0
    ):
        super().__init__()

        # Main transformation
        self.dense = nn.Linear(
            input_dim,
            units
        )

        self.batch_norm = nn.BatchNorm1d(units)

        self.relu = nn.ReLU()

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Residual shortcut
        if input_dim != units:
            self.shortcut = nn.Linear(
                input_dim,
                units
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward propagation through residual block.
        """

        shortcut = self.shortcut(x)

        x = self.dense(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)

        x = x + shortcut
        x = self.relu(x)

        return x


# ==================================================
# WeightNet
# ==================================================

class WeightNet(nn.Module):
    """
    Dynamic weighting network for SynDWE.

    Architecture
    ------------
    Input features
         |
    Linear(input_dim, 128)
         |
    ReLU -> BatchNorm -> Dropout(0.3)
         |
    ResidualBlock(128, 128, dropout=0.2)
         |
    ResidualBlock(128, 64, dropout=0.2)
         |
    ResidualBlock(64, 32, dropout=0.1)
         |
    Linear(32, n_models)
         |
       Softmax
         |
    Dynamic model weights

    Parameters
    ----------
    input_dim : int
        Number of input features.

    n_models : int, default=N_BASE_MODELS
        Number of base classifiers.

    Returns
    -------
    weights : torch.Tensor
        Sample-specific weight matrix.

        Shape:
            (batch_size, n_models)

        Each row sums to 1.
    """

    def __init__(
        self,
        input_dim,
        n_models=N_BASE_MODELS
    ):
        super().__init__()

        # ------------------------------------------
        # Input projection
        # ------------------------------------------

        self.input_layer = nn.Linear(
            input_dim,
            WEIGHTNET_INPUT_HIDDEN
        )

        self.input_batch_norm = nn.BatchNorm1d(
            WEIGHTNET_INPUT_HIDDEN
        )

        self.input_dropout = nn.Dropout(
            WEIGHTNET_INPUT_DROPOUT
        )

        self.relu = nn.ReLU()

        # ------------------------------------------
        # Residual blocks
        # ------------------------------------------

        self.residual_block_1 = ResidualBlock(
            input_dim=WEIGHTNET_INPUT_HIDDEN,
            units=WEIGHTNET_RESIDUAL_UNITS[0],
            dropout_rate=WEIGHTNET_RESIDUAL_DROPOUT[0]
        )

        self.residual_block_2 = ResidualBlock(
            input_dim=WEIGHTNET_RESIDUAL_UNITS[0],
            units=WEIGHTNET_RESIDUAL_UNITS[1],
            dropout_rate=WEIGHTNET_RESIDUAL_DROPOUT[1]
        )

        self.residual_block_3 = ResidualBlock(
            input_dim=WEIGHTNET_RESIDUAL_UNITS[1],
            units=WEIGHTNET_RESIDUAL_UNITS[2],
            dropout_rate=WEIGHTNET_RESIDUAL_DROPOUT[2]
        )

        # ------------------------------------------
        # Output layer
        # ------------------------------------------

        self.output_layer = nn.Linear(
            WEIGHTNET_RESIDUAL_UNITS[2],
            n_models
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Generate dynamic weights for base models.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.

            Shape:
                (batch_size, input_dim)

        Returns
        -------
        weights : torch.Tensor
            Dynamic weights for each base classifier.

            Shape:
                (batch_size, n_models)
        """

        # Input projection
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.input_batch_norm(x)
        x = self.input_dropout(x)

        # Residual feature transformations
        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.residual_block_3(x)

        # Generate model weights
        x = self.output_layer(x)

        weights = self.softmax(x)

        return weights