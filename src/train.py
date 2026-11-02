"""
SynDWE: Model Training

Cross-validation reproduces the original SynDWE
implementation, including:

- Global StandardScaler before five-fold CV
- KFold with random_state=42
- Inner five-fold OOF predictions
- WeightNet target weight construction
- Validation AUC-based early stopping
- Five-fold metric calculation

NOTE:
Global standardization and selecting WeightNet
using the evaluation fold can introduce optimistic
validation estimates. These steps are retained
only for reproducing the original implementation.
"""

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)

from src.config import (
    TRAIN_FILE,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    N_SPLITS,
    INNER_N_SPLITS,
    WEIGHTNET_MAX_EPOCHS,
    WEIGHTNET_BATCH_SIZE,
    WEIGHTNET_LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MIN_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    METRIC_NAMES
)

from src.data_utils import (
    set_random_seed,
    read_csv_dataset,
)

from src.base_models import build_base_models
from src.weightnet import WeightNet
from src.losses import categorical_cross_entropy

from src.ensemble import (
    generate_oof_predictions,
    build_pseudo_labels,
    dynamic_weighted_predict,
)


# ==================================================
# Original WeightNet training
# ==================================================

def train_weight_net(
    X_train,
    pseudo_labels,
    X_val,
    y_val,
    meta_val,
    input_dim,
    n_models,
    device,
    epochs=100,
    batch_size=32
):
    """
    Train WeightNet with original AUC early stopping.
    """

    model = WeightNet(
        input_dim=input_dim,
        n_models=n_models
    ).to(device)

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    pseudo_tensor = torch.tensor(
        pseudo_labels,
        dtype=torch.float32
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=device
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        pseudo_tensor
    )

    drop_last = len(train_dataset) % batch_size == 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=50,
        min_lr=1e-6
    )

    early_stop_patience = 45
    early_stop_counter = 0
    min_delta = 1e-6

    best_val_auc = -np.inf
    best_epoch = 0
    stop_epoch = epochs

    best_model_state = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(1, epochs + 1):

        # ------------------------------------------
        # Training
        # ------------------------------------------

        model.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predicted_weights = model(batch_X)

            loss = categorical_cross_entropy(
                predicted_weights,
                batch_y
            )

            loss.backward()
            optimizer.step()

            batch_size_current = batch_X.size(0)

            train_loss_sum += (
                loss.item() * batch_size_current
            )

            train_sample_count += batch_size_current

        train_loss = (
            train_loss_sum / train_sample_count
        )

        # ------------------------------------------
        # Validation AUC
        # ------------------------------------------

        model.eval()

        with torch.no_grad():

            val_weights = (
                model(X_val_tensor)
                .cpu()
                .numpy()
            )

        val_prob = np.sum(
            val_weights * meta_val,
            axis=1
        )

        val_auc = roc_auc_score(
            y_val,
            val_prob
        )

        scheduler.step(val_auc)

        # ------------------------------------------
        # Original early stopping
        # ------------------------------------------

        if val_auc > best_val_auc + min_delta:

            best_val_auc = val_auc
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            early_stop_counter = 0

            print(
                "[WeightNet] Epoch {:03d}: "
                "TrainLoss={:.6f}, "
                "ValAUC improved to {:.6f}".format(
                    epoch,
                    train_loss,
                    val_auc
                )
            )

        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:

            stop_epoch = epoch

            print(
                "[WeightNet] Early stopping at "
                "epoch {}; best epoch={}, "
                "best ValAUC={:.6f}".format(
                    stop_epoch,
                    best_epoch,
                    best_val_auc
                )
            )

            break

    else:

        print(
            "[WeightNet] Reached maximum epoch {}; "
            "best epoch={}, best ValAUC={:.6f}".format(
                epochs,
                best_epoch,
                best_val_auc
            )
        )

    model.load_state_dict(best_model_state)

    return model, best_epoch, best_val_auc, stop_epoch


# ==================================================
# Original five-fold cross-validation
# ==================================================

def cross_validate(
    train_file=TRAIN_FILE,
    results_dir=RESULTS_DIR
):
    """
    Perform the original five-fold cross-validation.

    Global standardization is intentionally
    preserved to match the original script.
    """

    set_random_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print(
        "Dynamic Weight Stacking "
        "(PyTorch, 5-Fold Cross-Validation)"
    )
    print("=" * 70)

    print("PyTorch device: {}".format(device))

    # ----------------------------------------------
    # Read training dataset
    # ----------------------------------------------

    X_all, y_all, feature_names = read_csv_dataset(
        train_file
    )

    # ----------------------------------------------
    # ORIGINAL GLOBAL STANDARDIZATION
    # ----------------------------------------------

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # ----------------------------------------------
    # Build base models
    # ----------------------------------------------

    base_models = build_base_models()
    n_models = len(base_models)

    # ----------------------------------------------
    # Outer five-fold CV
    # ----------------------------------------------

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(X_all),
        start=1
    ):

        print("\n===== Fold {} =====".format(fold_idx))

        X_tr = X_all[train_idx]
        X_val = X_all[val_idx]

        y_tr = y_all[train_idx]
        y_val = y_all[val_idx]

        # ------------------------------------------
        # Inner five-fold OOF predictions
        # ------------------------------------------

        meta_tr, meta_val = generate_oof_predictions(
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_val,
            base_models=base_models,
            n_splits=5,
            random_seed=42
        )

        # ------------------------------------------
        # Original target weight construction
        # ------------------------------------------

        pseudo = build_pseudo_labels(
            meta_train=meta_tr,
            y_train=y_tr
        )

        # ------------------------------------------
        # Original WeightNet training
        # ------------------------------------------

        (
            weight_model,
            best_epoch,
            best_val_auc,
            stop_epoch
        ) = train_weight_net(
            X_train=X_tr,
            pseudo_labels=pseudo,
            X_val=X_val,
            y_val=y_val,
            meta_val=meta_val,
            input_dim=X_tr.shape[1],
            n_models=n_models,
            device=device,
            epochs=100,
            batch_size=32
        )

        # ------------------------------------------
        # Original validation prediction
        # ------------------------------------------

        weight_model.eval()

        X_val_tensor = torch.tensor(
            X_val,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():

            val_weights = (
                weight_model(X_val_tensor)
                .cpu()
                .numpy()
            )

        val_prob = np.sum(
            val_weights * meta_val,
            axis=1
        )

        val_pred = (
            val_prob > 0.5
        ).astype(int)

        # ------------------------------------------
        # Evaluation metrics
        # ------------------------------------------

        # Accuracy

        val_acc = accuracy_score(
            y_val,
            val_pred
        )

        # Confusion matrix
        # TN: True Negative
        # FP: False Positive
        # FN: False Negative
        # TP: True Positive

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            val_pred,
            labels=[0, 1]
        ).ravel()

        # Sensitivity (SN)

        val_sn = (
            tp / (tp + fn)
            if (tp + fn) > 0 else np.nan
        )

        # Specificity (SP)

        val_sp = (
            tn / (tn + fp)
            if (tn + fp) > 0 else np.nan
        )

        # AUC

        val_auc = roc_auc_score(
            y_val,
            val_prob
        )

        # AUPR

        val_aupr = average_precision_score(
            y_val,
            val_prob
        )

        # F1-score

        val_f1 = f1_score(
            y_val,
            val_pred
        )

        # MCC

        val_mcc = matthews_corrcoef(
            y_val,
            val_pred
        )

        # ------------------------------------------
        # Print metrics
        # ------------------------------------------

        print(
            "[VAL] "
            "ACC={:.4f} "
            "SN={:.4f} "
            "SP={:.4f} "
            "AUC={:.4f} "
            "AUPR={:.4f} "
            "F1={:.4f} "
            "MCC={:.4f}".format(
                val_acc,
                val_sn,
                val_sp,
                val_auc,
                val_aupr,
                val_f1,
                val_mcc
            )
        )

        # ------------------------------------------
        # Save fold metrics
        # ------------------------------------------

        fold_metrics.append([
            val_acc,
            val_sn,
            val_sp,
            val_auc,
            val_aupr,
            val_f1,
            val_mcc
        ])

        # ------------------------------------------
        # Original resource cleanup
        # ------------------------------------------

        del weight_model
        del X_val_tensor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------------------------
    # Original five-fold statistics
    # ----------------------------------------------

    fold_metrics = np.asarray(fold_metrics)

    avg_metrics = fold_metrics.mean(axis=0)
    std_metrics = fold_metrics.std(axis=0)

    metric_names = METRIC_NAMES

    print("\n===== 5-Fold Validation Metrics =====")

    for metric_idx, metric_name in enumerate(
        metric_names
    ):

        print(
            "{:<4} : {:.4f} ± {:.4f}".format(
                metric_name,
                avg_metrics[metric_idx],
                std_metrics[metric_idx]
            )
        )

    # ----------------------------------------------
    # Save results to the requested directory
    # ----------------------------------------------

    results_dir = Path(results_dir)

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df = pd.DataFrame(
        fold_metrics,
        columns=metric_names
    )

    metrics_df.insert(
        0,
        "fold",
        np.arange(1, len(metrics_df) + 1)
    )

    metrics_file = (
        results_dir /
        "cross_validation_metrics.csv"
    )

    metrics_df.to_csv(
        str(metrics_file),
        index=False
    )

    print(
        "\nCross-validation metrics saved to: {}".format(
            metrics_file
        )
    )

    return metrics_df


# ==================================================
# Final WeightNet training without early stopping
# ==================================================

def train_weight_net_fixed_epochs(
    X_train,
    pseudo_labels,
    device,
    epochs=26,
    batch_size=32
):
    """
    Train final WeightNet for a fixed number
    of epochs, following the final-training code.
    """

    model = WeightNet(
        input_dim=X_train.shape[1],
        n_models=pseudo_labels.shape[1]
    ).to(device)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(pseudo_labels, dtype=torch.float32)
    )

    drop_last = len(train_dataset) % batch_size == 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-4
    )

    for epoch in range(1, epochs + 1):

        model.train()

        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predicted_weights = model(batch_X)

            loss = categorical_cross_entropy(
                predicted_weights,
                batch_y
            )

            loss.backward()
            optimizer.step()

            n = batch_X.size(0)

            train_loss_sum += loss.item() * n
            train_sample_count += n

        train_loss = (
            train_loss_sum / train_sample_count
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == epochs
        ):
            print(
                "[WeightNet] Epoch {:03d}/{}: "
                "TrainLoss={:.6f}".format(
                    epoch,
                    epochs,
                    train_loss
                )
            )

    model.eval()
    return model


# ==================================================
# Train and save final ensemble
# ==================================================

def train_final_model(
    train_file=TRAIN_FILE,
    save_dir=CHECKPOINT_DIR,
    epochs=26
):
    """
    Train the final SynDWE model.

    Note:
    This function follows the modular project's
    five-fold base-model saving strategy.
    It is separate from the CV reproduction logic.
    """

    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, feature_names = read_csv_dataset(
        train_file
    )

    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    base_models = build_base_models()

    meta_train, _, fitted_models = (
        generate_oof_predictions(
            X_train=X_all,
            y_train=y_all,
            X_val=None,
            base_models=base_models,
            n_splits=INNER_N_SPLITS,
            random_seed=RANDOM_SEED,
            return_fitted_models=True
        )
    )

    pseudo = build_pseudo_labels(
        meta_train,
        y_all
    )

    weight_model = train_weight_net_fixed_epochs(
        X_train=X_all,
        pseudo_labels=pseudo,
        device=device,
        epochs=epochs,
        batch_size=WEIGHTNET_BATCH_SIZE
    )

    model_names = [
        name for name, _ in base_models
    ]

    bundle_path = (
        save_dir / "stacking_bundle.joblib"
    )

    joblib.dump(
        {
            "scaler": scaler,
            "base_models": fitted_models,
            "model_names": model_names,
            "feature_names": feature_names,
            "label_name": "LABEL"
        },
        str(bundle_path)
    )

    weightnet_path = save_dir / "weightnet.pth"

    torch.save(
        {
            "model_state_dict": weight_model.state_dict(),
            "input_dim": X_all.shape[1],
            "n_models": len(base_models),
            "model_names": model_names,
            "epochs": epochs,
            "random_seed": RANDOM_SEED
        },
        str(weightnet_path)
    )

    metadata = {
        "model_name": "SynDWE",
        "random_seed": RANDOM_SEED,
        "feature_count": X_all.shape[1],
        "feature_names": feature_names,
        "base_model_names": model_names,
        "oof_splits": INNER_N_SPLITS,
        "weightnet_epochs": epochs,
        "batch_size": WEIGHTNET_BATCH_SIZE,
        "learning_rate": WEIGHTNET_LEARNING_RATE,
        "classification_threshold": 0.5,
        "base_model_strategy": "five_fold_average",
        "scaler": "StandardScaler",
        "bundle_file": bundle_path.name,
        "weightnet_file": weightnet_path.name
    }

    metadata_path = save_dir / "metadata.json"

    with open(
        str(metadata_path),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\n===== Final Model Saved =====")
    print("Base models: {}".format(bundle_path))
    print("WeightNet: {}".format(weightnet_path))
    print("Metadata: {}".format(metadata_path))

    return {
        "bundle_path": bundle_path,
        "weightnet_path": weightnet_path,
        "metadata_path": metadata_path
    }