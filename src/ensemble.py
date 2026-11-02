
"""
SynDWE: Dynamic Weighted Ensemble

Original five-fold OOF prediction and
dynamic weighting logic.
"""

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import KFold

from src.config import (
    RANDOM_SEED,
    INNER_N_SPLITS,
    CLASSIFICATION_THRESHOLD,
)


def generate_oof_predictions(
    X_train,
    y_train,
    X_val,
    base_models,
    n_splits=INNER_N_SPLITS,
    random_seed=RANDOM_SEED,
    return_fitted_models=False
):
    """
    Generate original OOF training predictions
    and averaged validation probabilities.
    """

    n_train = X_train.shape[0]
    n_models = len(base_models)

    meta_train = np.zeros(
        (n_train, n_models),
        dtype=np.float32
    )

    meta_val = None

    if X_val is not None:
        meta_val = np.zeros(
            (X_val.shape[0], n_models),
            dtype=np.float32
        )

    inner_kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_seed
    )

    fitted_models = {}

    for model_idx, (name, base_model) in enumerate(
        base_models
    ):

        print("Training base model: {}".format(name))

        if X_val is not None:
            val_folds = np.zeros(
                (X_val.shape[0], inner_kf.n_splits),
                dtype=np.float32
            )

        fold_models = []

        for inner_fold, (
            inner_train_idx,
            inner_oof_idx
        ) in enumerate(inner_kf.split(X_train)):

            model_clone = clone(base_model)

            model_clone.fit(
                X_train[inner_train_idx],
                y_train[inner_train_idx]
            )

            # Original OOF assignment
            meta_train[inner_oof_idx, model_idx] = (
                model_clone.predict_proba(
                    X_train[inner_oof_idx]
                )[:, 1]
            )

            # Original validation prediction assignment
            if X_val is not None:
                val_folds[:, inner_fold] = (
                    model_clone.predict_proba(
                        X_val
                    )[:, 1]
                )

            if return_fitted_models:
                fold_models.append(model_clone)

        if X_val is not None:
            meta_val[:, model_idx] = (
                val_folds.mean(axis=1)
            )

        if return_fitted_models:
            fitted_models[name] = fold_models

    if return_fitted_models:
        return meta_train, meta_val, fitted_models

    return meta_train, meta_val


def build_pseudo_labels(
    meta_train,
    y_train,
    threshold=CLASSIFICATION_THRESHOLD
):
    """
    Construct target weight distributions
    exactly as in the original implementation.
    """

    meta_bin = (
        meta_train > threshold
    ).astype(int)

    pseudo = (
        meta_bin == y_train.reshape(-1, 1)
    ).astype(np.float32)

    row_sum = pseudo.sum(
        axis=1,
        keepdims=True
    )

    zero_row = row_sum.squeeze() == 0

    # Original fallback behavior:
    # use predicted probabilities when
    # all base classifiers are incorrect.
    if zero_row.any():

        pseudo[zero_row] = meta_train[zero_row]

        row_sum = pseudo.sum(
            axis=1,
            keepdims=True
        )

    pseudo /= row_sum + 1e-12

    return pseudo


def dynamic_weighted_predict(
    model_probabilities,
    model_weights,
    threshold=CLASSIFICATION_THRESHOLD
):
    """
    Compute final dynamic ensemble probabilities.
    """

    final_probabilities = np.sum(
        model_weights * model_probabilities,
        axis=1
    )

    final_predictions = (
        final_probabilities > threshold
    ).astype(int)

    return final_probabilities, final_predictions


def predict_base_model_ensemble(
    X,
    fitted_models,
    model_names=None
):
    """
    Predict using saved five-fold base models.

    The prediction averaging order follows
    the original validation prediction code.
    """

    if model_names is None:
        model_names = [
            "xgb",
            "rf",
            "lgb",
            "gbdt"
        ]

    probabilities = np.zeros(
        (X.shape[0], len(model_names)),
        dtype=np.float32
    )

    for model_idx, name in enumerate(model_names):

        models = fitted_models[name]

        fold_predictions = np.zeros(
            (X.shape[0], len(models)),
            dtype=np.float32
        )

        for fold_idx, model in enumerate(models):
            fold_predictions[:, fold_idx] = (
                model.predict_proba(X)[:, 1]
            )

        probabilities[:, model_idx] = (
            fold_predictions.mean(axis=1)
        )

    return probabilities