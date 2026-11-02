"""
SynDWE: Independent Test Set Evaluation

This module provides functions for:

1. Loading saved SynDWE models
2. Predicting independent test datasets
3. Computing classification metrics
4. Saving prediction results

Expected input:
    100 feature columns + optional LABEL

The sequence_id column is not required.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
)

from src.config import (
    CHECKPOINT_DIR,
    RESULTS_DIR,
    CLASSIFICATION_THRESHOLD,
    METRIC_NAMES,
)

from src.data_utils import (
    read_csv_dataset,
    transform_features,
)

from src.weightnet import WeightNet

from src.ensemble import (
    predict_base_model_ensemble,
    dynamic_weighted_predict,
)


# ==================================================
# Load trained SynDWE model
# ==================================================

def load_syndwe_model(
    model_dir=CHECKPOINT_DIR,
    device=None,
):
    """
    Load the saved SynDWE ensemble.

    Returns
    -------
    scaler : StandardScaler
        Fitted training-data scaler.

    fitted_models : dict
        Saved base classifier fold ensembles.

    weight_model : WeightNet
        Trained dynamic weighting network.

    feature_names : list
        Ordered training feature names.

    model_names : list
        Base classifier order.

    device : torch.device
        Device used for WeightNet inference.
    """

    model_dir = Path(model_dir)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    bundle_path = model_dir / "stacking_bundle.joblib"
    weightnet_path = model_dir / "weightnet.pth"
    metadata_path = model_dir / "metadata.json"

    # ----------------------------------------------
    # Check model files
    # ----------------------------------------------

    for path in [
        bundle_path,
        weightnet_path,
        metadata_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required model file not found: {path}"
            )

    # ----------------------------------------------
    # Load metadata
    # ----------------------------------------------

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    # ----------------------------------------------
    # Load scaler and base models
    # ----------------------------------------------

    # Only load checkpoints from trusted sources.
    bundle = joblib.load(bundle_path)

    scaler = bundle["scaler"]
    fitted_models = bundle["base_models"]
    model_names = bundle["model_names"]
    feature_names = bundle["feature_names"]

    # ----------------------------------------------
    # Load WeightNet checkpoint
    # ----------------------------------------------

    # Only load checkpoints produced by this project.
    checkpoint = torch.load(
        weightnet_path,
        map_location=device
    )

    input_dim = checkpoint["input_dim"]
    n_models = checkpoint["n_models"]

    if len(feature_names) != input_dim:
        raise ValueError(
            "Feature dimension does not match "
            "the saved WeightNet input dimension."
        )

    if len(model_names) != n_models:
        raise ValueError(
            "Base-model count does not match "
            "the saved WeightNet output dimension."
        )

    if (
        metadata.get("base_model_names")
        != model_names
    ):
        raise ValueError(
            "Base-model order differs from metadata."
        )

    weight_model = WeightNet(
        input_dim=input_dim,
        n_models=n_models,
    ).to(device)

    weight_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    weight_model.eval()

    print("=" * 60)
    print("SynDWE model loaded successfully")
    print("=" * 60)
    print(f"Model directory: {model_dir}")
    print(f"Device: {device}")
    print(f"Features: {input_dim}")
    print(f"Base models: {model_names}")

    return (
        scaler,
        fitted_models,
        weight_model,
        feature_names,
        model_names,
        device,
    )


# ==================================================
# Predict WeightNet weights
# ==================================================

def predict_dynamic_weights(
    weight_model,
    X,
    device,
    batch_size=1024,
):
    """
    Generate sample-specific dynamic weights.

    Returns
    -------
    weights : numpy.ndarray
        Shape (n_samples, n_models).
    """

    weight_model.eval()

    all_weights = []

    with torch.no_grad():

        for start in range(0, len(X), batch_size):

            batch_X = torch.as_tensor(
                X[start:start + batch_size],
                dtype=torch.float32,
                device=device,
            )

            batch_weights = weight_model(
                batch_X
            )

            all_weights.append(
                batch_weights.cpu().numpy()
            )

    return np.concatenate(
        all_weights,
        axis=0,
    )


# ==================================================
# Calculate evaluation metrics
# ==================================================

def calculate_test_metrics(
    y_true,
    y_prob,
    threshold=CLASSIFICATION_THRESHOLD,
):
    """
    Calculate binary classification metrics.

    Metrics:
        ACC  : Accuracy
        SN   : Sensitivity
        SP   : Specificity
        AUC  : ROC AUC
        AUPR : Area under Precision-Recall curve
        F1   : F1-score
        MCC  : Matthews correlation coefficient

    ROC AUC and AUPR are undefined when the
    test set contains only one class.

    Sensitivity or specificity is also NaN when
    its corresponding class is absent.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # ----------------------------------------------
    # Binary predictions
    # ----------------------------------------------

    y_pred = (
        y_prob > threshold
    ).astype(np.int64)

    # ----------------------------------------------
    # Confusion matrix
    # ----------------------------------------------

    # Confusion matrix order:
    # TN: True Negative
    # FP: False Positive
    # FN: False Negative
    # TP: True Positive

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # ----------------------------------------------
    # Sensitivity (SN)
    # ----------------------------------------------

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0 else np.nan
    )

    # ----------------------------------------------
    # Specificity (SP)
    # ----------------------------------------------

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0 else np.nan
    )

    # ----------------------------------------------
    # ACC, SN, SP, F1, MCC
    # ----------------------------------------------

    metrics = {

        "ACC": float(
            accuracy_score(y_true, y_pred)
        ),

        "SN": float(sensitivity),

        "SP": float(specificity),

        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "MCC": float(
            matthews_corrcoef(
                y_true,
                y_pred
            )
        ),
    }

    # ----------------------------------------------
    # AUC and AUPR
    # ----------------------------------------------

    if len(np.unique(y_true)) == 2:

        metrics["AUC"] = float(
            roc_auc_score(
                y_true,
                y_prob
            )
        )

        metrics["AUPR"] = float(
            average_precision_score(
                y_true,
                y_prob
            )
        )

    else:

        metrics["AUC"] = np.nan
        metrics["AUPR"] = np.nan

    # ----------------------------------------------
    # Return metrics in a unified order
    # ----------------------------------------------

    return {
        metric_name: metrics[metric_name]
        for metric_name in METRIC_NAMES
    }


# ==================================================
# Independent test prediction
# ==================================================

def evaluate_test_set(
    test_file,
    model_dir=CHECKPOINT_DIR,
    output_dir=RESULTS_DIR,
    dataset_name="test",
    device=None,
):
    """
    Evaluate SynDWE on an independent test dataset.

    The input CSV may include LABEL, but does
    not need sequence_id.

    If LABEL exists, metrics are calculated.

    Returns
    -------
    results_df : pandas.DataFrame
        Prediction results.

    metrics : dict or None
        Evaluation metrics when labels exist.
    """

    # ----------------------------------------------
    # Load saved models
    # ----------------------------------------------

    (
        scaler,
        fitted_models,
        weight_model,
        feature_names,
        model_names,
        device,
    ) = load_syndwe_model(
        model_dir=model_dir,
        device=device,
    )

    # ----------------------------------------------
    # Read independent test dataset
    # ----------------------------------------------

    X_test, y_test, _ = read_csv_dataset(
        test_file,
        require_labels=False,
        expected_features=feature_names,
    )

    print(f"\nEvaluating dataset: {dataset_name}")
    print(f"Number of samples: {len(X_test)}")

    # ----------------------------------------------
    # Apply training-data standardization
    # ----------------------------------------------

    X_test_scaled = transform_features(
        X_test,
        scaler,
    )

    # ----------------------------------------------
    # Obtain base-model probabilities
    # ----------------------------------------------

    meta_test = predict_base_model_ensemble(
        X=X_test_scaled,
        fitted_models=fitted_models,
        model_names=model_names,
    )

    # ----------------------------------------------
    # Generate dynamic weights
    # ----------------------------------------------

    test_weights = predict_dynamic_weights(
        weight_model=weight_model,
        X=X_test_scaled,
        device=device,
    )

    # ----------------------------------------------
    # Dynamic ensemble prediction
    # ----------------------------------------------

    final_prob, final_pred = dynamic_weighted_predict(
        model_probabilities=meta_test,
        model_weights=test_weights,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    # ----------------------------------------------
    # Construct prediction results
    # ----------------------------------------------

    results_df = pd.DataFrame({
        "sample_index": np.arange(len(X_test)),
        "SynDWE_probability": final_prob,
        "SynDWE_prediction": final_pred,
    })

    if y_test is not None:
        results_df.insert(
            1,
            "LABEL",
            y_test,
        )

    # ----------------------------------------------
    # Base-model probabilities
    # ----------------------------------------------

    for idx, name in enumerate(model_names):

        results_df[f"{name}_probability"] = (
            meta_test[:, idx]
        )

    # ----------------------------------------------
    # WeightNet dynamic weights
    # ----------------------------------------------

    for idx, name in enumerate(model_names):

        results_df[f"{name}_weight"] = (
            test_weights[:, idx]
        )

    # ----------------------------------------------
    # Create output directory
    # ----------------------------------------------

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------
    # Save prediction results
    # ----------------------------------------------

    prediction_file = (
        output_dir / f"{dataset_name}_predictions.csv"
    )

    results_df.to_csv(
        prediction_file,
        index=False,
    )

    print(f"\nPredictions saved to: {prediction_file}")

    # ----------------------------------------------
    # Calculate metrics if labels are available
    # ----------------------------------------------

    metrics = None

    if y_test is not None:

        metrics = calculate_test_metrics(
            y_true=y_test,
            y_prob=final_prob,
        )

        print("\n===== Test Set Metrics =====")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # ------------------------------------------
        # Save metrics
        # ------------------------------------------

        metrics_df = pd.DataFrame([
            {
                "dataset": dataset_name,
                "n_samples": len(y_test),
                "n_positive": int(np.sum(y_test == 1)),
                "n_negative": int(np.sum(y_test == 0)),
                **metrics,
            }
        ])

        metrics_file = (
            output_dir / f"{dataset_name}_metrics.csv"
        )

        metrics_df.to_csv(
            metrics_file,
            index=False,
        )

        print(f"Metrics saved to: {metrics_file}")

    else:

        print(
            "LABEL is not available; "
            "prediction results were saved "
            "without calculating metrics."
        )

    return results_df, metrics