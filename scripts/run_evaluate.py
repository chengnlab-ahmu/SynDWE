
"""
Run SynDWE independent test evaluation.

Examples
--------
python -m scripts.run_evaluate --dataset ghs
python -m scripts.run_evaluate --dataset svdb
python -m scripts.run_evaluate --dataset ck
"""

import argparse

from src.evaluate import evaluate_test_set


DATASET_FILES = {
    "ghs": "data/ghs.csv",
    "svdb": "data/svdb.csv",
    "ck": "data/ck.csv",
}


def main():

    parser = argparse.ArgumentParser(
        description="SynDWE independent test evaluation"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ghs", "svdb", "ck"],
        required=True,
        help="Independent test dataset name",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="checkpoints",
        help="Directory containing saved models",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for evaluation results",
    )

    args = parser.parse_args()

    test_file = DATASET_FILES[args.dataset]

    evaluate_test_set(
        test_file=test_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    main()