# SynDWE: A Dynamic Weighted Ensemble Framework for Predicting Cancer-Driver Synonymous Mutations

SynDWE is a dynamic weighted ensemble framework for predicting cancer-driver synonymous mutations using **100 precomputed features**. It combines four base classifiers—XGBoost, Random Forest (RF), LightGBM, and Gradient Boosting Decision Trees (GBDT)—with a residual neural network (**WeightNet**) that assigns sample-specific weights to their prediction probabilities.

This repository provides modular Python code for cross-validation, final-model training, model serialization, and independent test-set evaluation.

> **Scope:** The code operates on already prepared, numeric, 100-feature CSV datasets.
## Project structure

```text
SynDWE/
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths, seeds, hyperparameters
│   ├── data_utils.py       # CSV loading, validation and standardization
│   ├── base_models.py      # XGBoost, RF, LightGBM, GBDT
│   ├── weightnet.py        # Residual WeightNet architecture
│   ├── losses.py           # Categorical cross-entropy loss
│   ├── ensemble.py         # Inner-fold OOF predictions and dynamic fusion
│   ├── train.py            # Cross-validation and final model training
│   └── evaluate.py         # Saved-model loading and test prediction
├── scripts/
│   ├── __init__.py
│   ├── run_cv.py           # Cross-validation entry point
│   ├── run_train.py        # Final-training entry point
│   ├── run_evaluate.py     # Test-evaluation entry point
│   └── remove_sequence_id.py  # Optional ID-removal utility
├── data/                   # Supply locally prepared CSV files
├── checkpoints/            # Saved model artifacts (generated)
├── results/                # Evaluation outputs (generated)
├── README.md
├── requirements.txt
└── .gitignore
```

The `remove_sequence_id.py` script is optional; place the previously provided ID-removal script at the path shown above if you want it included in the repository.

## Installation

Python **3.10 or newer** is recommended. From the project root:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch installation may require platform-specific instructions, especially for CUDA-enabled systems. Refer to the official PyTorch installation selector for a compatible build.

## Input data

Use **CSV files containing exactly 100 numeric feature columns** and, for supervised training or labeled evaluation, a binary `LABEL` column (0 = passenger candidate; 1 = driver candidate). For unlabeled inference, `LABEL` may be absent.

An example of the **column layout** (ellipses are illustrative, *not* literal CSV entries):

```text
AF,CADD_RAWSCORE,SYNVEP_SCORE_AVG,...,XLMR_788,LABEL
0.03623,0.27305,0.16724,...,0.56250,1
```

- `sequence_id` is **not required**. If present, the data loader removes it from model inputs.
- Column names must match those used for training; during evaluation, the loader automatically reorders columns to the saved feature order.
- Missing (`NaN`), infinite, nonnumeric, and extra/missing feature values are rejected. Complete imputation and feature selection **before** running this repository.
- The current implementation applies a `StandardScaler` fitted on training data and saves it with the model. This is distinct from the Min–Max normalization described in the manuscript's upstream feature-engineering section; verify the exact preprocessing protocol when reproducing reported results.
- For external data, do not recompute a new scaler using test-set values.

Place your **locally permitted** feature files at:

```text
data/train.csv
data/ghs.csv
data/svdb.csv
data/ck.csv
```

File names may be changed in the scripts. Do not upload restricted data merely to match these paths.

### Optional: remove `sequence_id`

If you saved the previously provided utility to `scripts/remove_sequence_id.py`, update its configured input/output paths and run it from the project root:

```bash
python scripts/remove_sequence_id.py
```

Removing an identifier alone does **not** guarantee that a dataset is de-identified or redistributable. Review licensing, provenance, consent/privacy obligations and possible linkage risks before publication.

## Methods overview

1. **Base learners:** XGBoost, RF, LightGBM and GBDT generate class-1 probabilities.
2. **OOF training signals:** Five inner folds produce out-of-fold (OOF) training probabilities. Each training sample is predicted by a learner that did not fit on that sample.
3. **Target weights:** Correctly classifying learners receive equal normalized target weights. If all four are incorrect, the original implementation normalizes their prediction probabilities.
4. **WeightNet:** A residual network (input projection to 128 hidden units; residual blocks with 128, 64 and 32 units; four-way Softmax) is trained on the target-weight distributions using categorical cross-entropy.
5. **Prediction:** The final probability is the sample-specific weighted sum of the four base-model probabilities; a probability **strictly greater than 0.5** is classified as positive.

## Five-fold cross-validation

```bash
python -m scripts.run_cv \
    --train data/train.csv \
    --output results/
```

This executes outer five-fold cross-validation, with five-fold inner OOF predictions per outer training fold. WeightNet uses a maximum of **100 epochs** and validation-AUC-based early stopping (patience **45**); the learning-rate scheduler uses patience **50**. The training script writes:

```text
results/cross_validation_metrics.csv
results/weightnet_auc_early_stopping_records.csv
```

The evaluation metrics include accuracy (ACC), sensitivity (SN), specificity (SP), area under the ROC curve (AUC), area under the precision–recall curve (AUPR), F1-score (F1), and Matthews correlation coefficient (MCC).

**Evaluation caveat:** The provided cross-validation implementation uses each outer validation fold to select the best WeightNet epoch **and** to report that fold's scores.

## Train and save the final model

```bash
python -m scripts.run_train \
    --train data/train.csv \
    --output checkpoints/ \
    --epochs 26
```

The current final-training implementation trains WeightNet for **26 fixed epochs** (from the previously provided final-training script), generates inner five-fold OOF signals, and saves **five fold-fitted estimators per base classifier** for averaged inference. This differs from the older full-data final-training script that refit one estimator per classifier; use model files produced by this repository's `run_train.py` with its `evaluate.py`.

Generated artifacts:

```text
checkpoints/stacking_bundle.joblib  # Scaler, fold models, feature names
checkpoints/weightnet.pth           # WeightNet state_dict and dimensions
checkpoints/metadata.json           # Model configuration and feature schema
```

The `26`-epoch choice is not automatically selected from cross-validation. For publication-quality reproducibility, document how the final training schedule was chosen.

## Evaluate independent datasets

First train a model, then run one of the following:

```bash
python -m scripts.run_evaluate --dataset ghs
python -m scripts.run_evaluate --dataset svdb
python -m scripts.run_evaluate --dataset ck
```

The `run_evaluate.py` entry point maps dataset names to the corresponding files under `data/`. It loads saved feature order and preprocessing, computes the four base-learner probabilities, predicts WeightNet's sample-specific weights, and writes:

```text
results/ghs_predictions.csv
results/ghs_metrics.csv   # Only if LABEL is present
```

Substitute `svdb` or `ck` for the corresponding test set. Prediction files include the SynDWE probability and binary label, four base probabilities, four dynamic weights, and the original `LABEL` when provided. `sample_index` is a generated row number, **not** a genomic variant ID.

For datasets with only one observed class, the implementation reports AUC and AUPR as undefined (`NaN`), rather than assigning them a misleading value. For an all-positive CK set, report the positive detection count / sensitivity as appropriate.

## Reproducibility and limitations

- Random seed: **42**; stochastic CPU/GPU operations and package versions can still influence exact results.
- Feature dimension: **100**; all feature columns must be precomputed consistently.
- The example implementation uses `KFold` rather than gene-group-aware CV. The manuscript's **GHS** is a separately constructed gene-holdout test set; do not confuse it with random five-fold CV.
- The repository does not include original feature-generation procedures or restricted underlying data.
- To reproduce manuscript results, verify the feature-generation, imputation, preprocessing, dataset-split, epoch-selection and saved-model strategy used in the reported experiments.
- Model files should only be loaded from **trusted sources**: `joblib.load()` may execute code from a malicious serialized file.

## Data availability

The original variant-level data and/or some annotations may have redistribution restrictions. Please use the repository with appropriately authorized, precomputed feature tables. Removing `sequence_id` is not sufficient on its own to establish whether data can legally or safely be shared. Add a dataset availability statement and source-specific license/usage terms before publishing any example data.

## Citation

If you use SynDWE, please cite the associated manuscript:

> **SynDWE: A Dynamic Weighted Ensemble Framework Integrating Multi-Source Features for Predicting Cancer-Driver Synonymous Mutations.**

## License

**To be determined by the repository maintainers.** Add an explicit `LICENSE` file before declaring the code open source. Third-party data, pretrained models, and annotations may have separate licenses.
