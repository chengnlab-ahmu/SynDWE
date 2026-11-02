
"""
SynDWE: Base Classifiers

This module defines four heterogeneous base classifiers
used in the SynDWE dynamic weighted ensemble framework:

    1. XGBoost
    2. Random Forest (RF)
    3. LightGBM
    4. Gradient Boosting Decision Tree (GBDT)

The hyperparameters follow the original SynDWE
implementation.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_SEED


# ==================================================
# Base model construction
# ==================================================

def build_base_models(random_seed=RANDOM_SEED):
    """
    Construct the four SynDWE base classifiers.

    Parameters
    ----------
    random_seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    base_models : list of tuple
        A list of (model_name, estimator) pairs.

        Model order:
            1. xgb
            2. rf
            3. lgb
            4. gbdt

    Notes
    -----
    The order must remain consistent with the
    WeightNet output dimensions.
    """

    base_models = [

        # ------------------------------------------
        # 1. XGBoost
        # ------------------------------------------

        (
            "xgb",
            XGBClassifier(
                n_estimators=200,
                max_depth=30,
                min_child_weight=15,
                gamma=0.5,
                subsample=0.6,
                colsample_bytree=0.6,
                reg_alpha=1.0,
                reg_lambda=2.0,
                eval_metric="logloss",
                random_state=random_seed,
            ),
        ),

        # ------------------------------------------
        # 2. Random Forest
        # ------------------------------------------

        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=60,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=random_seed,
            ),
        ),

        # ------------------------------------------
        # 3. LightGBM
        # ------------------------------------------

        (
            "lgb",
            LGBMClassifier(
                n_estimators=500,
                max_depth=25,
                num_leaves=256,
                min_data_in_leaf=50,
                learning_rate=0.02,
                feature_fraction=0.6,
                bagging_fraction=0.6,
                bagging_freq=1,
                lambda_l1=1.0,
                lambda_l2=2.0,
                random_state=random_seed,
                verbose=-1,
            ),
        ),

        # ------------------------------------------
        # 4. Gradient Boosting Decision Tree
        # ------------------------------------------

        (
            "gbdt",
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=random_seed,
            ),
        ),
    ]

    return base_models