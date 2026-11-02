
"""
SynDWE: Five-Fold Cross-Validation

Run five-fold cross-validation with:
    - Four base classifiers
    - Inner five-fold OOF predictions
    - Dynamic WeightNet training
    - AUC-based early stopping
    - Evaluation metrics

Usage
-----
python -m scripts.run_cv

python -m scripts.run_cv \
    --train data/train_top100.csv \
    --output results/
"""

import argparse

from src.config import TRAIN_FILE, RESULTS_DIR
from src.train import cross_validate


def main():

    parser = argparse.ArgumentParser(
        description="SynDWE five-fold cross-validation"
    )

    parser.add_argument(
        "--train",
        type=str,
        default=str(TRAIN_FILE),
        help="Path to the training CSV file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR),
        help="Directory for cross-validation results"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SynDWE: Five-Fold Cross-Validation")
    print("=" * 70)

    cross_validate(
        train_file=args.train,
        results_dir=args.output,
    )


if __name__ == "__main__":
    main()