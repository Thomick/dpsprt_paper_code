#!/usr/bin/env python3
"""
Generate box plots for SPRT comparison experiments.

This script reads results from experiment JSON files and generates box plots
to compare different SPRT algorithm variants across problem instances, alpha values,
and epsilon values.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

# Assume utils.py is in the same directory
from utils import ensure_dir_exists


# Define the algorithm styles (same as in plot_performance_metrics.py)
ALGORITHM_STYLES = {
    "ClassicalSPRT": {"color": "#1f77b4", "label": "Classical SPRT", "marker": "o"},
    "DP-SPRT": {"color": "#ff7f0e", "label": "DP-SPRT Laplace", "marker": "s"},
    "DP-SPRT-Subsampled": {
        "color": "#9467bd",
        "label": "DP-SPRT Laplace-Sub",
        "marker": "p",
    },
    "DP-SPRT-Gaussian": {
        "color": "#800080",
        "label": "DP-SPRT Gaussian",
        "marker": "*",
    },
    "Priv-SPRT": {"color": "#d62728", "label": "PrivSPRT", "marker": "D"},
}

# Problem instances as defined in run_all.py with simplified labels
PROBLEM_INSTANCES = [
    {
        "name": "easy_p0.3_p1.7",
        "mu0": "0.3",
        "mu1": "0.7",
        "label": "Instance 1",  # Easy
        "description": "Easy (μ₀=0.3, μ₁=0.7)",
    },
    {
        "name": "difficult_p0.45_p1.55",
        "mu0": "0.45",
        "mu1": "0.55",
        "label": "Instance 2",  # Hard 1
        "description": "Hard (μ₀=0.45, μ₁=0.55)",
    },
    {
        "name": "difficult_p0.05_p1.25",
        "mu0": "0.05",
        "mu1": "0.25",
        "label": "Instance 3",  # Hard 2
        "description": "Hard (μ₀=0.05, μ₁=0.25)",
    },
]


def extract_stopping_times(
    results_data: Dict[str, Dict[str, Any]],
    algo_name: str,
    alpha_val: float,
    epsilon_val: float,
    all_epsilons_config_list: List[float],
    hypothesis: str,
) -> Optional[List[float]]:
    """Extract stopping times for a specific algorithm, alpha, epsilon, and hypothesis."""
    try:
        if str(alpha_val) not in results_data.get(algo_name, {}):
            return None

        results_for_alpha = results_data[algo_name][str(alpha_val)]

        epsilon_idx = None
        try:
            epsilon_idx = all_epsilons_config_list.index(epsilon_val)
        except ValueError:
            return None

        if epsilon_idx is None or epsilon_idx >= len(results_for_alpha):
            return None

        data_point = results_for_alpha[epsilon_idx]

        if hypothesis not in data_point:
            return None

        stopping_times = data_point[hypothesis].get("stopping_times", None)

        if stopping_times is None or not isinstance(stopping_times, list):
            return None

        return stopping_times
    except Exception as e:
        print(
            f"Error extracting stopping times for {algo_name}, α={alpha_val}, ε={epsilon_val}: {e}"
        )
        return None


def load_results_data(results_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load results data and config from a JSON file."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return {}, {}
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON from results file: {results_file}")
        return {}, {}

    actual_results_data = data.get("results_data")
    if actual_results_data is None:
        actual_results_data = data.get("results", {})

    actual_config = data.get("config")
    if actual_config is None or not actual_config:
        actual_config = data.get("parameters", {})

    return actual_results_data, actual_config


def plot_boxplots_across_instances(
    base_instance_results_dir: str,
    output_dir: str,
    fixed_alpha: float = 0.05,
    fixed_epsilon: float = 1.0,
    hypothesis: str = "H0",
) -> None:
    """
    Generate box plots comparing algorithms across different problem instances.

    Args:
        base_instance_results_dir: Base directory containing results for different instances
        output_dir: Directory to save the box plots
        fixed_alpha: Alpha value to use for the comparison
        fixed_epsilon: Epsilon value to use for the comparison
        hypothesis: Which hypothesis to plot (H0 or H1)
    """
    base_dir = Path(base_instance_results_dir)
    plots_dir = Path(output_dir) / "boxplots"
    ensure_dir_exists(plots_dir)

    # Prepare data structures to collect stopping times for each algorithm and instance
    all_stopping_times = {}
    all_config_epsilons = None

    # First, load data from all instances
    for instance in PROBLEM_INSTANCES:
        instance_name = instance["name"]
        instance_dir = base_dir / instance_name
        instance_results_file = instance_dir / "sprt_comparison_results.json"

        if not instance_results_file.exists():
            print(f"⚠️ Results file not found for instance {instance_name}, skipping")
            continue

        results_data, config = load_results_data(str(instance_results_file))

        if not results_data:
            print(f"⚠️ No results data found for instance {instance_name}, skipping")
            continue

        # Get the epsilon list from config
        if all_config_epsilons is None:
            all_config_epsilons = sorted([float(e) for e in config.get("epsilons", [])])

        # For each algorithm, extract stopping times
        for algo_name in ALGORITHM_STYLES.keys():
            if algo_name not in all_stopping_times:
                all_stopping_times[algo_name] = {}

            # Extract stopping times for this algorithm and instance
            stopping_times = extract_stopping_times(
                results_data,
                algo_name,
                fixed_alpha,
                fixed_epsilon,
                all_config_epsilons,
                hypothesis,
            )

            if stopping_times:
                all_stopping_times[algo_name][instance_name] = stopping_times

    # Order algorithms: Classical SPRT, Priv-SPRT, then DP-SPRT variants
    algo_order = [
        "ClassicalSPRT",
        "Priv-SPRT",
        "DP-SPRT",
        "DP-SPRT-Gaussian",
        "DP-SPRT-Subsampled",
    ]
    actual_algo_order = [algo for algo in algo_order if algo in all_stopping_times]

    if not actual_algo_order:
        print(
            "⚠️ No algorithms found with valid data, skipping instance comparison plot"
        )
        return

    # Create a new figure
    fig, ax = plt.subplots(figsize=(9, 5))  # Slightly smaller figure

    # Set font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 12
    LEGEND_SIZE = 12

    # Reorganize data by instance first, then algorithm
    positions = []
    box_data = []
    colors = []
    all_labels = []

    # Calculate width for each group and gaps
    num_instances = len(PROBLEM_INSTANCES)
    num_algos = len(actual_algo_order)
    bar_width = 0.8
    group_width = num_algos * bar_width
    # Reduce gap between groups and start closer to y-axis
    group_positions = np.arange(0, num_instances) * (group_width + 1) + group_width / 2

    # For each instance, plot all algorithms side by side
    for i, instance in enumerate(PROBLEM_INSTANCES):
        instance_name = instance["name"]
        instance_label = instance["label"]
        instance_position = group_positions[i]

        # For each algorithm, add a box to this instance group
        for j, algo_name in enumerate(actual_algo_order):
            if (
                algo_name in all_stopping_times
                and instance_name in all_stopping_times[algo_name]
            ):
                box_data.append(all_stopping_times[algo_name][instance_name])
                box_position = instance_position + j * bar_width
                positions.append(box_position)
                colors.append(ALGORITHM_STYLES[algo_name]["color"])
                all_labels.append(f"{ALGORITHM_STYLES[algo_name]['label']}")

    # Create custom box plot with specific positions
    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=bar_width * 0.8,
    )

    # Set colors for each box
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.7)

    # Set x-axis labels at the center of each group (centered on groups of boxes)
    ax.set_xticks(
        [pos + (num_algos * bar_width) / 2 - bar_width / 2 for pos in group_positions]
    )
    ax.set_xticklabels([instance["label"] for instance in PROBLEM_INSTANCES])

    # Set plot properties
    ax.set_title(
        f"Stopping Times Distribution ({hypothesis}) (α={fixed_alpha}, ε={fixed_epsilon})",
        fontsize=TITLE_SIZE,
    )
    ax.set_ylabel(f"Stopping Time (Log Scale)", fontsize=LABEL_SIZE)
    # Adjust x-axis limits to better utilize space
    ax.set_xlim(
        left=group_positions[0] - group_width / 3,
        right=max(group_positions) + group_width,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Set y-scale to logarithmic
    ax.set_yscale("log")

    # Set tick font sizes
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # Create a separate legend for the algorithms
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=ALGORITHM_STYLES[algo]["color"], alpha=0.7
        )
        for algo in actual_algo_order
    ]
    ax.legend(
        legend_handles,
        [ALGORITHM_STYLES[algo]["label"] for algo in actual_algo_order],
        loc="upper right",
        fontsize=LEGEND_SIZE,
    )

    # Save the plot
    plot_filename = f"boxplot_instances_comparison_{hypothesis}_alpha_{fixed_alpha}_eps_{fixed_epsilon}.pdf"
    plt.tight_layout()
    plt.savefig(plots_dir / plot_filename, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {plots_dir / plot_filename}")


def plot_boxplots_across_alphas_for_instance(
    instance_results_file: str,
    output_dir: str,
    instance_name: str,
    fixed_epsilon: float = 1.0,
    selected_alphas: List[float] = None,
    hypothesis: str = "H0",
) -> None:
    """
    Generate box plots comparing algorithms across different alpha values for a specific instance.

    Args:
        instance_results_file: Path to the instance results JSON file
        output_dir: Directory to save the box plots
        instance_name: Name of the instance
        fixed_epsilon: Epsilon value to use for the comparison
        selected_alphas: List of alpha values to include (max 3)
        hypothesis: Which hypothesis to plot (H0 or H1)
    """
    plots_dir = Path(output_dir) / "boxplots"
    ensure_dir_exists(plots_dir)

    # Load results data
    results_data, config = load_results_data(instance_results_file)

    if not results_data:
        print(
            f"❌ No results data found in {instance_results_file}, skipping alpha comparison plot for {instance_name}"
        )
        return

    # Get the list of alphas and epsilons from config
    all_config_alphas = sorted([float(a) for a in config.get("alphas_betas", [])])
    all_config_epsilons = sorted([float(e) for e in config.get("epsilons", [])])

    if not all_config_alphas or not all_config_epsilons:
        print(f"❌ Alphas or epsilons not found in config for {instance_name}")
        return

    # If no specific alphas are provided, select at most 3 (low, medium, high)
    if not selected_alphas:
        if len(all_config_alphas) <= 3:
            selected_alphas = all_config_alphas
        else:
            # Select low, medium, and high alpha values
            indices = [0, 2, 4]
            selected_alphas = [all_config_alphas[i] for i in indices]
    else:
        # Ensure all selected alphas are in the config
        selected_alphas = [a for a in selected_alphas if a in all_config_alphas][:3]

    if not selected_alphas:
        print(f"❌ No valid alpha values selected for comparison in {instance_name}")
        return

    # Prepare data structures to collect stopping times for each algorithm and alpha
    all_stopping_times = {}

    # Order algorithms: Classical SPRT, Priv-SPRT, then DP-SPRT variants
    algo_order = [
        "ClassicalSPRT",
        "Priv-SPRT",
        "DP-SPRT",
        "DP-SPRT-Gaussian",
        "DP-SPRT-Subsampled",
    ]
    actual_algo_order = [algo for algo in algo_order if algo in results_data]

    # For each algorithm and alpha, extract stopping times
    for algo_name in actual_algo_order:
        if algo_name not in all_stopping_times:
            all_stopping_times[algo_name] = {}

        for alpha_val in selected_alphas:
            stopping_times = extract_stopping_times(
                results_data,
                algo_name,
                alpha_val,
                fixed_epsilon,
                all_config_epsilons,
                hypothesis,
            )

            if stopping_times:
                all_stopping_times[algo_name][alpha_val] = stopping_times

    if not all_stopping_times:
        print(
            f"❌ No valid stopping time data found for any algorithm in {instance_name}"
        )
        return

    # Create a new figure
    fig, ax = plt.subplots(figsize=(9, 5))  # Slightly smaller figure

    # Set font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 10

    # Reorganize data by alpha value first, then algorithm
    positions = []
    box_data = []
    colors = []

    # Calculate width for each group and gaps
    num_alphas = len(selected_alphas)
    num_algos = len(actual_algo_order)
    bar_width = 0.8
    group_width = num_algos * bar_width
    # Reduce gap between groups and start closer to y-axis
    group_positions = np.arange(0, num_alphas) * (group_width + 1) + group_width / 2

    # For each alpha, plot all algorithms side by side
    for i, alpha_val in enumerate(sorted(selected_alphas)):
        alpha_position = group_positions[i]

        # For each algorithm, add a box to this alpha group
        for j, algo_name in enumerate(actual_algo_order):
            if (
                algo_name in all_stopping_times
                and alpha_val in all_stopping_times[algo_name]
            ):
                box_data.append(all_stopping_times[algo_name][alpha_val])
                box_position = alpha_position + j * bar_width
                positions.append(box_position)
                colors.append(ALGORITHM_STYLES[algo_name]["color"])

    # Create custom box plot with specific positions
    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=bar_width * 0.8,
    )

    # Set colors for each box
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.7)

    # Set x-axis labels at the center of each group
    ax.set_xticks(
        [pos + (num_algos * bar_width) / 2 - bar_width / 2 for pos in group_positions]
    )
    ax.set_xticklabels([f"α = {alpha_val}" for alpha_val in sorted(selected_alphas)])

    # Create a separate legend for the algorithms
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=ALGORITHM_STYLES[algo]["color"], alpha=0.7
        )
        for algo in actual_algo_order
    ]
    ax.legend(
        legend_handles,
        [ALGORITHM_STYLES[algo]["label"] for algo in actual_algo_order],
        loc="upper right",
        fontsize=LEGEND_SIZE,
    )

    # Set plot properties
    instance_label = next(
        (inst["label"] for inst in PROBLEM_INSTANCES if inst["name"] == instance_name),
        instance_name,
    )
    ax.set_title(
        f"{instance_label}: Stopping Times ({hypothesis}) Across Alpha Values (ε={fixed_epsilon})",
        fontsize=TITLE_SIZE,
    )
    ax.set_ylabel(f"Stopping Time (Log Scale)", fontsize=LABEL_SIZE)
    # Adjust x-axis limits to better utilize space
    ax.set_xlim(
        left=group_positions[0] - group_width / 3,
        right=max(group_positions) + group_width,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Set y-scale to logarithmic
    ax.set_yscale("log")

    # Set tick font sizes
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # Save the plot
    plot_filename = (
        f"boxplot_{instance_name}_alpha_comparison_{hypothesis}_eps_{fixed_epsilon}.pdf"
    )
    plt.tight_layout()
    plt.savefig(plots_dir / plot_filename, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {plots_dir / plot_filename}")


def plot_boxplots_across_epsilons_for_instance(
    instance_results_file: str,
    output_dir: str,
    instance_name: str,
    fixed_alpha: float = 0.05,
    selected_epsilons: List[float] = None,
    hypothesis: str = "H0",
) -> None:
    """
    Generate box plots comparing algorithms across different epsilon values for a specific instance.

    Args:
        instance_results_file: Path to the instance results JSON file
        output_dir: Directory to save the box plots
        instance_name: Name of the instance
        fixed_alpha: Alpha value to use for the comparison
        selected_epsilons: List of epsilon values to include (max 3)
        hypothesis: Which hypothesis to plot (H0 or H1)
    """
    plots_dir = Path(output_dir) / "boxplots"
    ensure_dir_exists(plots_dir)

    # Load results data
    results_data, config = load_results_data(instance_results_file)

    if not results_data:
        print(
            f"❌ No results data found in {instance_results_file}, skipping epsilon comparison plot for {instance_name}"
        )
        return

    # Get the list of alphas and epsilons from config
    all_config_alphas = sorted([float(a) for a in config.get("alphas_betas", [])])
    all_config_epsilons = sorted([float(e) for e in config.get("epsilons", [])])

    if not all_config_alphas or not all_config_epsilons:
        print(f"❌ Alphas or epsilons not found in config for {instance_name}")
        return

    # If no specific epsilons are provided, select at most 3 (low, medium, high)
    if not selected_epsilons:
        if len(all_config_epsilons) <= 3:
            selected_epsilons = all_config_epsilons
        else:
            # Select low, medium, and high epsilon values (distributed in log-space)
            selected_epsilons = [
                all_config_epsilons[0],
                all_config_epsilons[1],
                all_config_epsilons[2],
            ]
    else:
        # Ensure all selected epsilons are in the config
        selected_epsilons = [e for e in selected_epsilons if e in all_config_epsilons][
            :3
        ]

    if not selected_epsilons:
        print(f"❌ No valid epsilon values selected for comparison in {instance_name}")
        return

    # Prepare data structures to collect stopping times for each algorithm and epsilon
    all_stopping_times = {}

    # Order algorithms: Classical SPRT, Priv-SPRT, then DP-SPRT variants
    algo_order = [
        "ClassicalSPRT",
        "Priv-SPRT",
        "DP-SPRT",
        "DP-SPRT-Gaussian",
        "DP-SPRT-Subsampled",
    ]
    actual_algo_order = [algo for algo in algo_order if algo in results_data]

    # For each algorithm and epsilon, extract stopping times
    for algo_name in actual_algo_order:
        if algo_name not in all_stopping_times:
            all_stopping_times[algo_name] = {}

        for epsilon_val in selected_epsilons:
            stopping_times = extract_stopping_times(
                results_data,
                algo_name,
                fixed_alpha,
                epsilon_val,
                all_config_epsilons,
                hypothesis,
            )

            if stopping_times:
                all_stopping_times[algo_name][epsilon_val] = stopping_times

    if not all_stopping_times:
        print(
            f"❌ No valid stopping time data found for any algorithm in {instance_name}"
        )
        return

    # Create a new figure
    fig, ax = plt.subplots(figsize=(9, 5))  # Slightly smaller figure

    # Set font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    LEGEND_SIZE = 10

    # Reorganize data by epsilon value first, then algorithm
    positions = []
    box_data = []
    colors = []

    # Calculate width for each group and gaps
    num_epsilons = len(selected_epsilons)
    num_algos = len(actual_algo_order)
    bar_width = 0.8
    group_width = num_algos * bar_width
    # Reduce gap between groups and start closer to y-axis
    group_positions = np.arange(0, num_epsilons) * (group_width + 1) + group_width / 2

    # For each epsilon, plot all algorithms side by side
    for i, epsilon_val in enumerate(sorted(selected_epsilons)):
        epsilon_position = group_positions[i]

        # For each algorithm, add a box to this epsilon group
        for j, algo_name in enumerate(actual_algo_order):
            if (
                algo_name in all_stopping_times
                and epsilon_val in all_stopping_times[algo_name]
            ):
                box_data.append(all_stopping_times[algo_name][epsilon_val])
                box_position = epsilon_position + j * bar_width
                positions.append(box_position)
                colors.append(ALGORITHM_STYLES[algo_name]["color"])

    # Create custom box plot with specific positions
    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        showfliers=False,
        widths=bar_width * 0.8,
    )

    # Set colors for each box
    for box, color in zip(bp["boxes"], colors):
        box.set(facecolor=color, alpha=0.7)

    # Set x-axis labels at the center of each group
    ax.set_xticks(
        [pos + (num_algos * bar_width) / 2 - bar_width / 2 for pos in group_positions]
    )
    ax.set_xticklabels(
        [f"ε = {epsilon_val}" for epsilon_val in sorted(selected_epsilons)]
    )

    # Create a separate legend for the algorithms
    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=ALGORITHM_STYLES[algo]["color"], alpha=0.7
        )
        for algo in actual_algo_order
    ]
    ax.legend(
        legend_handles,
        [ALGORITHM_STYLES[algo]["label"] for algo in actual_algo_order],
        loc="upper right",
        fontsize=LEGEND_SIZE,
    )

    # Set plot properties
    instance_label = next(
        (inst["label"] for inst in PROBLEM_INSTANCES if inst["name"] == instance_name),
        instance_name,
    )
    ax.set_title(
        f"{instance_label}: Stopping Times ({hypothesis}) Across Epsilon Values (α={fixed_alpha})",
        fontsize=TITLE_SIZE,
    )
    ax.set_ylabel(f"Stopping Time (Log Scale)", fontsize=LABEL_SIZE)
    # Adjust x-axis limits to better utilize space
    ax.set_xlim(
        left=group_positions[0] - group_width / 3,
        right=max(group_positions) + group_width,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Set y-scale to logarithmic
    ax.set_yscale("log")

    # Set tick font sizes
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # Save the plot
    plot_filename = f"boxplot_{instance_name}_epsilon_comparison_{hypothesis}_alpha_{fixed_alpha}.pdf"
    plt.tight_layout()
    plt.savefig(plots_dir / plot_filename, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {plots_dir / plot_filename}")


def generate_all_boxplots(
    main_results_dir: str,
    instance_results_dir: str,
    output_dir: str = None,
    fixed_alpha: float = 0.05,
    fixed_epsilon: float = 1.0,
) -> None:
    """Generate all types of box plots."""
    if output_dir is None:
        output_dir = main_results_dir

    # 1. Generate cross-instance comparison plots
    print("📊 Generating box plots comparing algorithms across instances...")
    plot_boxplots_across_instances(
        instance_results_dir,
        output_dir,
        fixed_alpha=fixed_alpha,
        fixed_epsilon=fixed_epsilon,
        hypothesis="H0",
    )

    plot_boxplots_across_instances(
        instance_results_dir,
        output_dir,
        fixed_alpha=fixed_alpha,
        fixed_epsilon=fixed_epsilon,
        hypothesis="H1",
    )

    # 2. Generate per-instance comparison plots
    print("📊 Generating per-instance box plots...")
    instance_files_found = False

    for instance in PROBLEM_INSTANCES:
        instance_name = instance["name"]
        instance_dir = Path(instance_results_dir) / instance_name
        instance_results_file = instance_dir / "sprt_comparison_results.json"

        if not instance_results_file.exists():
            print(f"⚠️ Results file not found for instance {instance_name}, skipping")
            continue

        instance_files_found = True

        # Generate alpha comparison plots for this instance
        print(f"📊 Generating alpha comparison plots for instance {instance_name}...")
        plot_boxplots_across_alphas_for_instance(
            str(instance_results_file),
            output_dir,
            instance_name,
            fixed_epsilon=fixed_epsilon,
            hypothesis="H0",
        )

        plot_boxplots_across_alphas_for_instance(
            str(instance_results_file),
            output_dir,
            instance_name,
            fixed_epsilon=fixed_epsilon,
            hypothesis="H1",
        )

        # Generate epsilon comparison plots for this instance
        print(f"📊 Generating epsilon comparison plots for instance {instance_name}...")
        plot_boxplots_across_epsilons_for_instance(
            str(instance_results_file),
            output_dir,
            instance_name,
            fixed_alpha=fixed_alpha,
            hypothesis="H0",
        )

        plot_boxplots_across_epsilons_for_instance(
            str(instance_results_file),
            output_dir,
            instance_name,
            fixed_alpha=fixed_alpha,
            hypothesis="H1",
        )

    if not instance_files_found:
        print("⚠️ No instance result files were found for generating boxplots")

    print("✅ All box plots generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate box plots for SPRT comparison experiments"
    )
    parser.add_argument(
        "--main-results-dir",
        type=str,
        default="results",
        help="Directory containing the main SPRT comparison results",
    )
    parser.add_argument(
        "--instance-results-dir",
        type=str,
        default="results/problem_instances",
        help="Directory containing results for different problem instances",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the generated box plots (defaults to main-results-dir)",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=0.05,
        help="Alpha value to use for fixed-alpha comparisons",
    )
    parser.add_argument(
        "--fixed-epsilon",
        type=float,
        default=1.0,
        help="Epsilon value to use for fixed-epsilon comparisons",
    )

    args = parser.parse_args()

    generate_all_boxplots(
        args.main_results_dir,
        args.instance_results_dir,
        args.output_dir,
        args.fixed_alpha,
        args.fixed_epsilon,
    )
