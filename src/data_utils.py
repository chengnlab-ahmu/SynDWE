
"""
SynDWE: Data Utilities

Data loading and preprocessing following the
original SynDWE cross-validation implementation.

The sequence_id column is optional.
"""

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


def set_random_seed(seed=RANDOM_SEED):
    """
    Set random seeds following the original code.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def read_csv_dataset(
    file_path,
    require_labels=True,
    expected_features=None
):
    """
    Read the 100-feature SynDWE dataset.

    sequence_id is optional and never used
    as a model input.

    Returns
    -------
    X : numpy.ndarray
        Float32 feature matrix.
    y : numpy.ndarray or None
        Int64 binary labels.
    feature_names : list
        Feature names in original column order.
    """

    df = pd.read_csv(file_path)

    # Only change from the original ID handling:
    # sequence_id is no longer required.
    if "sequence_id" in df.columns:
        df = df.drop(columns=["sequence_id"])

    if df.columns.duplicated().any():
        raise ValueError("Duplicate feature columns found.")

    if require_labels and "LABEL" not in df.columns:
        raise ValueError("LABEL column is missing.")

    if "LABEL" in df.columns:
        if df.columns[-1] != "LABEL":
            raise ValueError(
                "LABEL must be the last column."
            )

        label_values = df["LABEL"]

        if label_values.isna().any():
            raise ValueError(
                "LABEL contains missing values."
            )

        y = label_values.to_numpy(dtype=np.int64)

        unique_labels, label_counts = np.unique(
            y,
            return_counts=True
        )

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                "LABEL must contain only 0 and 1."
            )

        feature_df = df.drop(columns=["LABEL"])

    else:
        y = None
        feature_df = df

    # Preserve original feature order.
    if expected_features is not None:
        expected_features = list(expected_features)

        if set(feature_df.columns) != set(expected_features):
            raise ValueError(
                "Feature columns do not match training data."
            )

        feature_df = feature_df[expected_features]

    feature_names = feature_df.columns.tolist()

    if len(feature_names) != 100:
        raise ValueError(
            "Expected 100 features, found {}.".format(
                len(feature_names)
            )
        )

    # Exactly as in the original implementation:
    X = feature_df.to_numpy(dtype=np.float32)

    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    print("样本数量：{}".format(X.shape[0]))
    print("特征数量：{}".format(X.shape[1]))

    if y is not None:
        unique_labels, label_counts = np.unique(
            y,
            return_counts=True
        )

        print(
            "标签分布：",
            dict(zip(unique_labels, label_counts))
        )

    return X, y, feature_names


def fit_standard_scaler(X_train):
    """
    Fit StandardScaler.

    IMPORTANT:
    Do not cast the transformed data to float32.

    This preserves the original code's behavior.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    return X_scaled, scaler


def transform_features(X, scaler):
    """
    Transform data using an existing scaler.
    """
    return scaler.transform(X)


def prepare_cv_fold(X, y, train_idx, val_idx):
    """
    Split an already-standardized matrix.

    For original-result reproduction, the scaler
    must be fitted before the outer KFold.
    """

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    return X_train, X_val, y_train, y_val