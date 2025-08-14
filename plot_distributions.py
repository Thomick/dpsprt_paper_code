"""
Script for plotting stopping time distributions from SPRT experiment results.

This script provides functions for visualizing the distributions of stopping times
for different SPRT algorithms under both H0 and H1 hypotheses.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

from utils import ensure_dir_exists
from plot_performance_metrics import plot_all_new_performance_plots


def plot_stopping_time_distributions(
    results_data: Dict[str, Dict[str, Any]],
    epsilons: List[float],
    alphas_betas: List[float],
    mu0: float,
    mu1: float,
    results_dir: str = "results",
) -> None:
    """Plot the distribution of stopping times for each algorithm.

    Args:
        results_data: Dictionary containing results
        epsilons: List of epsilon values
        alphas_betas: List of alpha=beta values
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        results_dir: Directory to save plots
    """
    # Create directory for plots
    plots_dir = Path(results_dir) / "plots" / "distributions"
    ensure_dir_exists(plots_dir)

    # Define algorithm styles
    algorithm_styles = {
        # "ClassicalSPRT": {"color": "#1f77b4", "label": "Classical SPRT"},
        "DP-SPRT": {"color": "#ff7f0e", "label": "DP-SPRT"},
        "TunedDP-SPRT": {"color": "#2ca02c", "label": "Tuned DP-SPRT"},
        "Priv-SPRT": {"color": "#d62728", "label": "PrivSPRT"},
        # "DP-SPRT-Subsampled": {"color": "#9467bd", "label": "DP-SPRT Subsampled"},
        # "DP-SPRT-Gaussian": {"color": "#800080", "label": "DP-SPRT Gaussian"},
    }

    # For each combination of epsilon and alpha_beta
    for epsilon in epsilons:
        for alpha_beta in alphas_betas:
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))

            # Plot H0 distributions
            ax = axes[0]
            for algorithm, style in algorithm_styles.items():
                if algorithm in results_data:
                    epsilon_idx = epsilons.index(epsilon)
                    # alpha_beta_idx = alphas_betas.index(alpha_beta) # Not used

                    # Use str(alpha_beta) for dictionary key access
                    if str(alpha_beta) in results_data[algorithm]:
                        # Access H0 results directly
                        results = results_data[algorithm][str(alpha_beta)][epsilon_idx][
                            "H0"
                        ]

                        # Extract stopping times
                        stopping_times = results["stopping_times"]

                        # Plot histogram
                        if (
                            isinstance(stopping_times, list) and stopping_times
                        ):  # Ensure it's a non-empty list
                            ax.hist(
                                stopping_times,
                                bins=50,
                                alpha=0.5,
                                color=style["color"],
                                label=f"{style['label']} (mean={np.mean(stopping_times):.1f})",
                            )

                            # Plot mean line
                            ax.axvline(
                                np.mean(stopping_times),
                                color=style["color"],
                                linestyle="--",
                                linewidth=1.5,
                            )

            ax.set_title(
                f"Stopping Time Distribution (H0, ε={epsilon}, α=β={alpha_beta})",
                fontsize=9,
            )
            ax.set_xlabel("Number of Samples", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(alpha=0.3)

            # Plot H1 distributions
            ax = axes[1]
            for algorithm, style in algorithm_styles.items():
                if algorithm in results_data:
                    epsilon_idx = epsilons.index(epsilon)
                    # alpha_beta_idx = alphas_betas.index(alpha_beta) # Not used

                    # Use str(alpha_beta) for dictionary key access
                    if str(alpha_beta) in results_data[algorithm]:
                        # Access H1 results directly
                        results = results_data[algorithm][str(alpha_beta)][epsilon_idx][
                            "H1"
                        ]

                        # Extract stopping times
                        stopping_times = results["stopping_times"]

                        # Plot histogram
                        if (
                            isinstance(stopping_times, list) and stopping_times
                        ):  # Ensure it's a non-empty list
                            ax.hist(
                                stopping_times,
                                bins=50,
                                alpha=0.5,
                                color=style["color"],
                                label=f"{style['label']} (mean={np.mean(stopping_times):.1f})",
                            )

                            # Plot mean line
                            ax.axvline(
                                np.mean(stopping_times),
                                color=style["color"],
                                linestyle="--",
                                linewidth=1.5,
                            )

            ax.set_title(
                f"Stopping Time Distribution (H1, ε={epsilon}, α=β={alpha_beta})",
                fontsize=9,
            )
            ax.set_xlabel("Number of Samples", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(axis="both", which="major", labelsize=7)
            ax.grid(alpha=0.3)

            """ plt.suptitle(
                f"Stopping Time Distributions (μ0={mu0}, μ1={mu1}, ε={epsilon}, α=β={alpha_beta})",
                fontsize=10,
            ) """
            plt.tight_layout(rect=[0, 0.03, 1, 0.94])

            # Save figure
            plot_path = (
                plots_dir / f"stopping_times_eps_{epsilon}_alpha_beta_{alpha_beta}.pdf"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight", format="pdf")
            plt.close(fig)
            print(f"Saved: {plot_path}")


def plot_aggregate_distributions(
    results_data: Dict[str, Dict[str, Any]],
    results_dir: str = "results",
) -> None:
    """Plot aggregate stopping time distributions for each algorithm.

    Args:
        results_data: Dictionary containing results
        results_dir: Directory to save plots
    """
    # Create directory for plots
    plots_dir = Path(results_dir) / "plots" / "distributions"
    ensure_dir_exists(plots_dir)

    # Define algorithm styles
    algorithm_styles = {
        "ClassicalSPRT": {"color": "#1f77b4", "label": "Classical SPRT"},
        "DP-SPRT": {"color": "#ff7f0e", "label": "DP-SPRT"},
        "TunedDP-SPRT": {"color": "#2ca02c", "label": "Tuned DP-SPRT"},
        "Priv-SPRT": {"color": "#d62728", "label": "PrivSPRT"},
        "DP-SPRT-Subsampled": {"color": "#9467bd", "label": "DP-SPRT Subsampled"},
        "DP-SPRT-Gaussian": {"color": "#800080", "label": "DP-SPRT Gaussian"},
    }

    # Get list of algorithms
    algorithms = list(results_data.keys())

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Plot H0 distributions
    ax = axes[0]
    for algorithm, style in algorithm_styles.items():
        if algorithm in algorithms:
            # Gather all stopping times for this algorithm
            all_stopping_times = []
            for alpha_beta in results_data[algorithm]:
                for result in results_data[algorithm][alpha_beta]:
                    # Directly access stopping times, assuming the new structure
                    if "H0" in result and "stopping_times" in result["H0"]:
                        h0_stopping_times = result["H0"]["stopping_times"]
                        all_stopping_times.extend(h0_stopping_times)

            if all_stopping_times:
                # Plot histogram
                ax.hist(
                    all_stopping_times,
                    bins=50,
                    alpha=0.5,
                    color=style["color"],
                    label=f"{style['label']} (mean={np.mean(all_stopping_times):.1f})",
                )

                # Plot mean line
                ax.axvline(
                    np.mean(all_stopping_times),
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.5,
                )

    ax.set_title("Aggregate Stopping Time Distribution (H0)", fontsize=9)
    ax.set_xlabel("Number of Samples", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.grid(alpha=0.3)

    # Plot H1 distributions
    ax = axes[1]
    for algorithm, style in algorithm_styles.items():
        if algorithm in algorithms:
            # Gather all stopping times for this algorithm
            all_stopping_times = []
            for alpha_beta in results_data[algorithm]:
                for result in results_data[algorithm][alpha_beta]:
                    # Directly access stopping times, assuming the new structure
                    if "H1" in result and "stopping_times" in result["H1"]:
                        h1_stopping_times = result["H1"]["stopping_times"]
                        all_stopping_times.extend(h1_stopping_times)

            if all_stopping_times:
                # Plot histogram
                ax.hist(
                    all_stopping_times,
                    bins=50,
                    alpha=0.5,
                    color=style["color"],
                    label=f"{style['label']} (mean={np.mean(all_stopping_times):.1f})",
                )

                # Plot mean line
                ax.axvline(
                    np.mean(all_stopping_times),
                    color=style["color"],
                    linestyle="--",
                    linewidth=1.5,
                )

    ax.set_title("Aggregate Stopping Time Distribution (H1)", fontsize=9)
    ax.set_xlabel("Number of Samples", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.grid(alpha=0.3)

    """ plt.suptitle(
        "Aggregate Stopping Time Distributions (All Configurations)", fontsize=10
    ) """
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # Save figure
    plot_path = plots_dir / "aggregate_stopping_times.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {plot_path}")


def plot_all_distributions(results_file: str, results_dir: str = "results") -> None:
    """Load results data and generate all plots including original distributions and new performance plots."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON from results file: {results_file}")
        return

    # Try to get data using new keys first, then fall back to older common keys
    actual_results_data = data.get("results_data")
    if actual_results_data is None:
        actual_results_data = data.get("results")  # Fallback for results data

    actual_config = data.get("config")
    if actual_config is None or not actual_config:  # Check if None or empty dict
        actual_config = data.get("parameters")  # Fallback for config data
        if actual_config:  # If fallback was successful, print a note
            print("ℹ️ Using 'parameters' key from JSON as configuration.")
        else:  # If still no config, create an empty one to avoid errors later, though warnings will persist
            actual_config = {}

    if not actual_results_data:
        print(
            "❌ Neither 'results_data' nor 'results' key found in JSON file. Cannot generate any plots."
        )
        return

    # Ensure actual_config is a dictionary for .get() calls later
    if not isinstance(actual_config, dict):
        print(
            f"❌ Expected 'config' or 'parameters' to be a dictionary, but got {type(actual_config)}. Plots may fail."
        )
        actual_config = {}  # Default to empty dict to prevent crashes with .get

    mu0 = actual_config.get("mu0", 0.3)
    mu1 = actual_config.get("mu1", 0.7)

    epsilons_str_or_float = actual_config.get("epsilons", [])
    alphas_betas_str_or_float = actual_config.get("alphas_betas", [])

    try:
        epsilons_float = sorted([float(e) for e in epsilons_str_or_float])
        alphas_betas_float = sorted([float(ab) for ab in alphas_betas_str_or_float])
    except ValueError:
        print(
            "❌ Error converting epsilons or alphas/betas from config to float. Skipping original distribution plots."
        )
        epsilons_float = []
        alphas_betas_float = []

    if not epsilons_float or not alphas_betas_float:
        print(
            "❌ 'epsilons' or 'alphas_betas' not found or invalid in config. Skipping original distribution plots."
        )
    else:
        print("\\\\n📊 Generating original stopping time distribution plots...")
        plot_stopping_time_distributions(
            actual_results_data,
            epsilons_float,
            alphas_betas_float,
            mu0,
            mu1,
            results_dir,
        )
        print("✅ Original distribution plots generated.")

        print("\\\\n📊 Generating aggregate stopping time distribution plots...")
        # Ensure plot_aggregate_distributions can handle the actual_results_data structure
        # It might need adjustment if it also assumes a specific sub-key for alpha/epsilon lists
        plot_aggregate_distributions(actual_results_data, results_dir)
        print("✅ Aggregate distribution plots generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot stopping time distributions and performance metrics from SPRT experiment results."
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/sprt_comparison/sprt_comparison_results.json",
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    plot_all_distributions(args.results_file, args.results_dir)
