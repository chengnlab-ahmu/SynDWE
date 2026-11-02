"""
SynDWE: Configuration

Centralized settings for model training,
cross-validation, and evaluation.
"""

from pathlib import Path

# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILE = DATA_DIR / "train.csv"

# ==================================================
# Reproducibility
# ==================================================

RANDOM_SEED = 42

# ==================================================
# Dataset configuration
# ==================================================

LABEL_COLUMN = "LABEL"
ID_COLUMN = "sequence_id"

N_FEATURES = 100
CLASSIFICATION_THRESHOLD = 0.5

# ==================================================
# Cross-validation
# ==================================================

N_SPLITS = 5
INNER_N_SPLITS = 5

# ==================================================
# WeightNet training
# ==================================================

WEIGHTNET_MAX_EPOCHS = 100
WEIGHTNET_BATCH_SIZE = 32
WEIGHTNET_LEARNING_RATE = 5e-4

# Learning rate scheduler
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 50
MIN_LEARNING_RATE = 1e-6

# AUC-based early stopping
EARLY_STOPPING_PATIENCE = 45
EARLY_STOPPING_MIN_DELTA = 1e-6

# ==================================================
# WeightNet architecture
# ==================================================

WEIGHTNET_INPUT_HIDDEN = 128

WEIGHTNET_RESIDUAL_UNITS = [128, 64, 32]

WEIGHTNET_INPUT_DROPOUT = 0.3

WEIGHTNET_RESIDUAL_DROPOUT = [
    0.2,
    0.2,
    0.1
]

# ==================================================
# Base classifiers
# ==================================================

BASE_MODEL_NAMES = [
    "xgb",
    "rf",
    "lgb",
    "gbdt"
]

N_BASE_MODELS = len(BASE_MODEL_NAMES)

# ==================================================
# Evaluation metrics
# ==================================================

METRIC_NAMES = [
    "ACC",
    "SN",
    "SP",
    "AUC",
    "AUPR",
    "F1",
    "MCC"
]