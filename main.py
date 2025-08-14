"""
Main script for running SPRT comparison experiments.

This script provides a command-line interface for running experiments
with different SPRT algorithms and analyzing the results.
"""

import argparse
import time
from pathlib import Path
import json
import numpy as np

from experiment import run_algorithm_comparison
from plot_comparison import create_summary_plots, plot_from_dataframe
from utils import ensure_dir_exists


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run SPRT comparison experiments.")
    parser.add_argument(
        "--mu0",
        type=float,
        default=0.3,
        help="Null hypothesis parameter (default: 0.3)",
    )
    parser.add_argument(
        "--mu1",
        type=float,
        default=0.7,
        help="Alternative hypothesis parameter (default: 0.7)",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0, 2.0, 5.0],
        help="List of privacy parameters to test (default: 0.1 0.5 1.0 2.0 5.0)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="List of alpha=beta values to test (default: 0.05 0.1 0.2)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        help="Privacy failure probability for PrivSPRT (default: 1e-5)",
    )
    parser.add_argument(
        "--A",
        type=float,
        default=1.0,
        help="Truncation parameter for PrivSPRT (default: 1.0)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=500,
        help="Number of trials per configuration (default: 500)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples per sequence (default: 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing results",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to existing results file (for --plot-only)",
    )

    return parser.parse_args()


def main():
    """Main function to run experiments and generate plots."""
    # Parse command-line arguments
    args = parse_args()

    # Create results directory
    results_dir = args.results_dir
    ensure_dir_exists(Path(results_dir))

    # Save command-line arguments
    args_file = Path(results_dir) / "args.json"
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2, default=lambda x: str(x))

    if args.plot_only:
        if args.results_file:
            print(f"Generating plots from existing results file: {args.results_file}")
            create_summary_plots(args.results_file, results_dir)
        else:
            # Check for default results file
            default_results_file = Path(results_dir) / "sprt_comparison_results.json"
            if default_results_file.exists():
                print(
                    f"Generating plots from existing results file: {default_results_file}"
                )
                create_summary_plots(str(default_results_file), results_dir)
            else:
                # Check for CSV file
                default_csv_file = Path(results_dir) / "sprt_comparison_results.csv"
                if default_csv_file.exists():
                    print(
                        f"Generating plots from existing CSV file: {default_csv_file}"
                    )
                    plot_from_dataframe(str(default_csv_file), results_dir)
                else:
                    print(
                        "Error: No results file found. Please specify --results-file or run experiments first."
                    )
                    return
    else:
        print("Running SPRT comparison experiments...")
        print(f"Parameters: mu0={args.mu0}, mu1={args.mu1}")
        print(f"Epsilons: {args.epsilons}")
        print(f"Alpha=Beta values: {args.alphas}")
        print(f"Number of trials: {args.num_trials}")
        print(f"Maximum samples: {args.max_samples}")
        print(f"Results will be saved to: {results_dir}")

        # Run experiments
        start_time = time.time()
        results_dict = run_algorithm_comparison(
            mu0=args.mu0,
            mu1=args.mu1,
            epsilons=args.epsilons,
            alphas_betas=args.alphas,
            delta=args.delta,
            A=args.A,
            num_trials=args.num_trials,
            max_samples=args.max_samples,
            seed=args.seed,
            use_tqdm=not args.no_tqdm,
            results_dir=results_dir,
        )
        end_time = time.time()

        print(f"Experiments completed in {end_time - start_time:.2f} seconds.")
        print(f"Generating plots...")

        # Generate plots
        results_file = Path(results_dir) / "sprt_comparison_results.json"
        create_summary_plots(str(results_file), results_dir)

        print(f"All done! Results and plots saved to {results_dir}")


if __name__ == "__main__":
    main()
