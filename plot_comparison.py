"""
Visualization module for SPRT comparison experiments.

This module provides functions for visualizing the results of SPRT comparison
experiments, including performance metrics plots and distribution plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any


def generate_comparison_plots(
    epsilons: List[float],
    alphas_betas: List[float],
    results_dict: Dict[str, Dict[float, List[Dict[str, float]]]],
    mu0: float,
    mu1: float,
    results_dir: str = "results",
) -> None:
    """
    Generate comparison plots for SPRT algorithm comparison.
    - Performance vs. Epsilon for each alpha_beta.
    - Performance vs. Alpha/Beta for each epsilon.

    Args:
        epsilons: List of epsilon values
        alphas_betas: List of alpha=beta values
        results_dict: Dictionary containing results
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        results_dir: Directory to save results
    """
    from utils import ensure_dir_exists

    algorithm_styles = {
        "ClassicalSPRT": {"color": "#1f77b4", "marker": "o", "label": "Classical SPRT"},
        "DP-SPRT": {"color": "#ff7f0e", "marker": "s", "label": "DP-SPRT"},
        "TunedDP-SPRT": {"color": "#2ca02c", "marker": "^", "label": "Tuned DP-SPRT"},
        "Priv-SPRT": {"color": "#d62728", "marker": "D", "label": "PrivSPRT"},
        "DP-SPRT-Gaussian": {
            "color": "#800080",
            "marker": "*",
            "label": "DP-SPRT Gaussian",
        },
    }
    valid_algorithm_names = [name for name in algorithm_styles if name in results_dict]

    base_plots_dir = Path(results_dir) / "plots"
    comparison_plots_dir = base_plots_dir / "comparison_plots"
    ensure_dir_exists(comparison_plots_dir)

    metrics_to_plot_defs = [
        ("avg_sample_size", "Average Sample Size", True),  # key, Y-label, is_log_scale
        ("type_i_rate", "Type I Error Rate", False),
        ("type_ii_rate", "Type II Error Rate", False),
    ]

    # --- Plot 1: Performance vs. Epsilon (for each alpha_beta) ---
    for alpha_beta_val in alphas_betas:
        fig, axs = plt.subplots(3, 1, figsize=(4, 7.5), sharex=True)
        fig.suptitle(
            f"Performance vs. Epsilon (α=β={alpha_beta_val}, μ0={mu0}, μ1={mu1})",
            fontsize=10,
            y=0.99,
        )

        for plot_idx, (metric_key, ylabel_base, y_log_scale) in enumerate(
            metrics_to_plot_defs
        ):
            ax = axs[plot_idx]
            for name in valid_algorithm_names:
                style = algorithm_styles[name]
                y_values = []
                for eps_idx, _ in enumerate(epsilons):
                    try:
                        if str(alpha_beta_val) in results_dict[name] and eps_idx < len(
                            results_dict[name][str(alpha_beta_val)]
                        ):
                            res = results_dict[name][str(alpha_beta_val)][eps_idx]
                            if "metrics" in res:
                                y_values.append(res["metrics"].get(metric_key, np.nan))
                            else:
                                y_values.append(np.nan)
                        else:
                            y_values.append(np.nan)
                    except KeyError:
                        y_values.append(np.nan)

                ax.plot(
                    epsilons,
                    y_values,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.5,
                    markersize=4,
                    linestyle="-",
                )

            ax.set_ylabel(ylabel_base, fontsize=8)
            ax.set_title(f"{ylabel_base} vs. Epsilon", fontsize=9)
            if y_log_scale:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.7)

            if plot_idx == 1:  # Type I error
                ax.axhline(
                    y=alpha_beta_val,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Target α={alpha_beta_val}",
                )
            if plot_idx == 2:  # Type II error
                ax.axhline(
                    y=alpha_beta_val,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Target β={alpha_beta_val}",
                )

            ax.legend(fontsize=7, loc="best")
            ax.tick_params(axis="both", which="major", labelsize=7)

        axs[-1].set_xlabel("Epsilon (ε)", fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.5)
        plot_path = (
            comparison_plots_dir
            / f"perf_vs_epsilon_alpha_beta_{alpha_beta_val:.2f}.pdf"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"Saved: {plot_path}")

    # --- Plot 2: Performance vs. Alpha/Beta (for each epsilon) ---
    for eps_idx, epsilon_val in enumerate(epsilons):
        fig, axs = plt.subplots(3, 1, figsize=(4, 7.5), sharex=True)
        fig.suptitle(
            f"Performance vs. Alpha/Beta (ε={epsilon_val}, μ0={mu0}, μ1={mu1})",
            fontsize=10,
            y=0.99,
        )

        for plot_idx, (metric_key, ylabel_base, y_log_scale) in enumerate(
            metrics_to_plot_defs
        ):
            ax = axs[plot_idx]
            for name in valid_algorithm_names:
                style = algorithm_styles[name]
                y_values = []
                for ab_val in alphas_betas:
                    try:
                        if str(ab_val) in results_dict[name] and eps_idx < len(
                            results_dict[name][str(ab_val)]
                        ):
                            res = results_dict[name][str(ab_val)][eps_idx]
                            if "metrics" in res:
                                y_values.append(res["metrics"].get(metric_key, np.nan))
                            else:
                                y_values.append(np.nan)
                        else:
                            y_values.append(np.nan)
                    except KeyError:
                        y_values.append(np.nan)

                ax.plot(
                    alphas_betas,
                    y_values,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.5,
                    markersize=4,
                    linestyle="-",
                )

            ax.set_ylabel(ylabel_base, fontsize=8)
            ax.set_title(f"{ylabel_base} vs. Alpha/Beta", fontsize=9)
            if y_log_scale:
                ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.7)

            if plot_idx == 1:  # Type I Error
                ax.plot(
                    alphas_betas,
                    alphas_betas,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    label="Target α = x",
                )
            if plot_idx == 2:  # Type II Error
                ax.plot(
                    alphas_betas,
                    alphas_betas,
                    color="black",
                    linestyle=":",
                    linewidth=1.5,
                    label="Target β = x",
                )

            ax.legend(fontsize=7, loc="best")
            ax.tick_params(axis="both", which="major", labelsize=7)

        axs[-1].set_xlabel("Alpha/Beta (α=β)", fontsize=8)
        axs[-1].set_xticks(
            alphas_betas
        )  # Ensure all alpha_beta values are shown as ticks
        axs[-1].set_xticklabels([f"{ab:.2f}" for ab in alphas_betas], fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.5)
        plot_path = (
            comparison_plots_dir / f"perf_vs_alpha_beta_epsilon_{epsilon_val:.1f}.pdf"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig)
        print(f"Saved: {plot_path}")

    print(f"All comparison plots saved to {comparison_plots_dir}")


def plot_from_dataframe(csv_file: str, results_dir: str = "results") -> None:
    """Generate plots from a results dataframe.

    Args:
        csv_file: Path to the CSV file containing results
        results_dir: Directory to save plots
    """
    # Load dataframe
    df = pd.read_csv(csv_file)

    # Extract parameters
    epsilons = sorted(df["epsilon"].unique())
    alphas_betas = sorted(df["alpha_beta"].unique())
    mu0 = df["mu0"].iloc[0]
    mu1 = df["mu1"].iloc[0]

    # Create a results dictionary in the format expected by generate_comparison_plots
    results_dict = {}

    # Find all algorithm columns
    algorithm_cols = {}
    for col in df.columns:
        if "_avg_sample_size" in col or "_type_i_rate" in col or "_type_ii_rate" in col:
            algorithm = col.split("_")[0]
            if algorithm not in algorithm_cols:
                algorithm_cols[algorithm] = []
            algorithm_cols[algorithm].append(col)

    # Build results dictionary
    for algorithm in algorithm_cols:
        results_dict[algorithm] = {}
        for alpha_beta in alphas_betas:
            results_dict[algorithm][alpha_beta] = []

            for epsilon in epsilons:
                filtered_df = df[
                    (df["alpha_beta"] == alpha_beta) & (df["epsilon"] == epsilon)
                ]
                if not filtered_df.empty:
                    result_entry = {}
                    for col in algorithm_cols[algorithm]:
                        metric_name = "_".join(
                            col.split("_")[1:]
                        )  # Extract metric name after algorithm name
                        result_entry[metric_name] = filtered_df[col].iloc[0]
                    results_dict[algorithm][alpha_beta].append(result_entry)

    # Generate plots
    generate_comparison_plots(
        epsilons, alphas_betas, results_dict, mu0, mu1, results_dir
    )


def plot_stopping_time_distributions(
    results_data: Dict[str, Dict[str, Any]],
    epsilon: float,
    alpha_beta: float,
    mu0: float,
    mu1: float,
    results_dir: str = "results",
) -> None:
    """Plot the distribution of stopping times for each algorithm.

    Args:
        results_data: Dictionary containing results
        epsilon: Epsilon value to plot
        alpha_beta: Alpha/beta value to plot
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        results_dir: Directory to save plots
    """
    from utils import ensure_dir_exists

    # Create directory for plots
    plots_dir = Path(results_dir) / "plots" / "distributions"
    ensure_dir_exists(plots_dir)

    # Define algorithm styles
    algorithm_styles = {
        "ClassicalSPRT": {"color": "#1f77b4", "label": "Classical SPRT"},
        "DP-SPRT": {"color": "#ff7f0e", "label": "DP-SPRT"},
        "TunedDP-SPRT": {"color": "#2ca02c", "label": "Tuned DP-SPRT"},
        "Priv-SPRT": {"color": "#d62728", "label": "PrivSPRT"},
        "DP-SPRT-Gaussian": {"color": "#800080", "label": "DP-SPRT Gaussian"},
    }

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # Plot H0 distributions
    ax = axes[0]
    for algorithm, style in algorithm_styles.items():
        if algorithm in results_data:
            results = results_data[algorithm]

            # Extract stopping times for H0
            stopping_times = results["raw_results"]["H0"]["stopping_times"]

            # Plot histogram
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
        f"Stopping Time Distribution (H0, ε={epsilon}, α=β={alpha_beta})", fontsize=9
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
            results = results_data[algorithm]

            # Extract stopping times for H1
            stopping_times = results["raw_results"]["H1"]["stopping_times"]

            # Plot histogram
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
        f"Stopping Time Distribution (H1, ε={epsilon}, α=β={alpha_beta})", fontsize=9
    )
    ax.set_xlabel("Number of Samples", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.legend(fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=7)
    ax.grid(alpha=0.3)

    plt.suptitle(
        f"Stopping Time Distributions (μ0={mu0}, μ1={mu1}, ε={epsilon}, α=β={alpha_beta})",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # Save figure
    plot_path = plots_dir / f"stopping_times_eps_{epsilon}_alpha_beta_{alpha_beta}.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"Saved: {plot_path}")


def create_summary_plots(
    results_file: str,
    results_dir: str = "results",
) -> None:
    """Create summary plots from a results file.

    Args:
        results_file: Path to the results JSON file
        results_dir: Directory to save plots
    """
    import json

    # Load results
    with open(results_file, "r") as f:
        data = json.load(f)

    # Extract parameters
    parameters = data["parameters"]
    mu0 = parameters["mu0"]
    mu1 = parameters["mu1"]
    epsilons = parameters["epsilons"]
    alphas_betas = parameters["alphas_betas"]
    results_dict = data["results"]

    # Generate comparison plots
    generate_comparison_plots(
        epsilons, alphas_betas, results_dict, mu0, mu1, results_dir
    )

    # Create CSV from results for easier analysis
    from utils import save_results_to_dataframe

    csv_file = Path(results_dir) / "sprt_comparison_results.csv"
    save_results_to_dataframe(
        epsilons,
        alphas_betas,
        list(results_dict.keys()),
        results_dict,
        mu0,
        mu1,
        filename=csv_file,
    )

    print(f"Results saved to CSV: {csv_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots from SPRT comparison results."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/sprt_comparison_results.json",
        help="Path to the results JSON file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    create_summary_plots(args.results_file, args.results_dir)
