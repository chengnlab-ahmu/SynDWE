
"""
SynDWE: Final Model Training

Train the final SynDWE model using the complete
training dataset.

Components
----------
1. Four base classifiers
2. Five-fold OOF predictions
3. WeightNet dynamic weight learning
4. Model checkpoint saving

Usage
-----
python -m scripts.run_train

python -m scripts.run_train \
    --train data/train.csv \
    --output checkpoints/ \
    --epochs 26
"""

import argparse

from src.config import TRAIN_FILE, CHECKPOINT_DIR
from src.train import train_final_model


def main():

    parser = argparse.ArgumentParser(
        description="Train the final SynDWE model"
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
        default=str(CHECKPOINT_DIR),
        help="Directory for saving trained models"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=26,
        help="Number of WeightNet training epochs"
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")

    print("=" * 70)
    print("SynDWE: Final Model Training")
    print("=" * 70)

    train_final_model(
        train_file=args.train,
        save_dir=args.output,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()