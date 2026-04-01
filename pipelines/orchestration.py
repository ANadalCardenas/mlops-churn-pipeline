"""
Single CLI entrypoint for the full ML pipeline. Imports and calls src/ modules
in the correct order:

1. Parses CLI arguments (--data-version, --experiment-name, --run-name,
   --output-dir)
2. Creates the output directory tree (output_dir/ and output_dir/figures/)
3. Delegates to train.run_training(...) to execute the complete pipeline

This is the script that CI (pr_validation.yml) calls on every PR.

CLI args:
    --data-version      (default: v1)
    --experiment-name   (default: churn-experiment)
    --run-name          (default: orchestration-run)
    --output-dir        (default: reports/run)
"""

import argparse
import os
import sys

# Make sure src/ is importable when the script is run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train import run_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--experiment-name", default="churn-experiment")
    parser.add_argument("--run-name", default="orchestration-run")
    parser.add_argument("--output-dir", default="reports/run")
    args = parser.parse_args()

    # Create the output directory tree before training so sub-modules can write freely
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    run_training(
        data_version=args.data_version,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
