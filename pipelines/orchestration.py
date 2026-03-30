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
