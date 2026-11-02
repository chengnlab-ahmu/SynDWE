
"""
SynDWE: Loss Functions

This module implements the categorical cross-entropy
loss used to train WeightNet.

The loss measures the difference between the
sample-specific dynamic weights predicted by WeightNet
and the target weight distributions constructed from
out-of-fold (OOF) predictions.
"""

import torch


# ==================================================
# Categorical Cross-Entropy Loss
# ==================================================

def categorical_cross_entropy(predictions, targets):
    """
    Compute categorical cross-entropy loss between
    predicted weights and target weight distributions.

    Parameters
    ----------
    predictions : torch.Tensor
        Dynamic weights predicted by WeightNet.

        Shape:
            (batch_size, n_models)

        Each row represents a probability distribution
        over the base classifiers.

    targets : torch.Tensor
        Target weight distributions derived from
        OOF predictions.

        Shape:
            (batch_size, n_models)

    Returns
    -------
    loss : torch.Tensor
        Mean categorical cross-entropy loss.

    Notes
    -----
    The loss is calculated as:

        L = -mean(sum(targets * log(predictions), dim=1))

    This implementation follows the original
    SynDWE training code.
    """

    # Avoid numerical instability caused by log(0)
    predictions = torch.clamp(
        predictions,
        min=1e-12,
        max=1.0
    )

    # Calculate cross-entropy for each sample
    loss = -torch.sum(
        targets * torch.log(predictions),
        dim=1
    )

    # Return mean loss across the batch
    return loss.mean()