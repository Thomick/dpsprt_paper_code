import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import json

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
from utils import ensure_dir_exists


def _get_metric_value(
    results_data: Dict[str, Dict[str, Any]],
    algo_name: str,
    alpha_val: float,
    epsilon_val: float,
    all_epsilons_config_list: List[float],
    metric_json_key: str,
    data_hypothesis_key: str,
    is_list_metric: bool,
) -> Optional[Union[float, Tuple[float, float, float]]]:
    """Helper function to extract a metric value and percentile intervals from the results data."""
    try:
        if str(alpha_val) not in results_data.get(algo_name, {}):
            return None
        results_for_alpha = results_data[algo_name][str(alpha_val)]

        epsilon_idx: Optional[int] = None
        try:
            epsilon_idx = all_epsilons_config_list.index(epsilon_val)
        except ValueError:
            return None

        if epsilon_idx is None or epsilon_idx >= len(results_for_alpha):
            return None

        data_point = results_for_alpha[epsilon_idx]

        if data_hypothesis_key not in data_point:
            return None

        parent_data_dict = data_point[data_hypothesis_key]
        metric_val = parent_data_dict.get(metric_json_key)

        if metric_val is None:
            return None

        if is_list_metric:
            if isinstance(metric_val, list) and len(metric_val) > 0:
                # Ensure all elements are numbers before calculating statistics
                if not all(isinstance(x, (int, float)) for x in metric_val):
                    return None
                # Calculate mean and 25th/75th percentiles
                mean_val = np.mean(metric_val)
                lower_percentile = np.percentile(metric_val, 25)
                upper_percentile = np.percentile(metric_val, 75)
                return mean_val, lower_percentile, upper_percentile
            return None
        else:
            try:
                return float(metric_val)
            except (ValueError, TypeError):
                return None
    except Exception:  # Catching a broad exception if any step fails
        return None


def plot_all_new_performance_plots(
    loaded_results_data: Dict[str, Any],
    loaded_config: Dict[str, Any],
    results_dir: str = "results",
) -> None:
    """Generates all the new performance plots for various metrics using pre-loaded data."""

    results_data = loaded_results_data
    config = loaded_config

    if not results_data:
        print("❌ (plot_all_new_performance_plots) No results data provided.")
        return

    all_config_alphas = sorted([float(a) for a in config.get("alphas_betas", [])])
    all_config_epsilons = sorted([float(e) for e in config.get("epsilons", [])])

    if not all_config_alphas or not all_config_epsilons:
        print(
            "❌ Alphas or epsilons not found in config. Cannot generate performance plots."
        )
        # Fallback for missing config
        all_config_epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        all_config_alphas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
        print(
            f"Warning: Using hardcoded alpha/epsilon lists due to missing config: {all_config_alphas}, {all_config_epsilons}"
        )

    metrics_to_plot_config = [
        {
            "id": "mst_h0",
            "name": "Mean Stopping Time (H0)",
            "json_key": "stopping_times",
            "hypothesis_key": "H0",
            "is_list_metric": True,
            "y_scale": "linear",
            "y_label": "Mean Stopping Time (H0)",
        },
        {
            "id": "mst_h1",
            "name": "Mean Stopping Time (H1)",
            "json_key": "stopping_times",
            "hypothesis_key": "H1",
            "is_list_metric": True,
            "y_scale": "linear",
            "y_label": "Mean Stopping Time (H1)",
        },
        {
            "id": "type1_error",
            "name": "Type 1 Error (Empirical)",
            "json_key": "type_i_rate",
            "hypothesis_key": "metrics",
            "is_list_metric": False,
            "y_scale": "linear",
            "y_label": "Type 1 Error (Empirical)",
        },
        {
            "id": "type2_error",
            "name": "Type 2 Error (Empirical)",
            "json_key": "type_ii_rate",
            "hypothesis_key": "metrics",
            "is_list_metric": False,
            "y_scale": "linear",
            "y_label": "Type 2 Error (Empirical)",
        },
    ]

    for metric_conf in metrics_to_plot_config:
        plot_id = metric_conf["id"]
        plot_name = metric_conf["name"]
        json_key = metric_conf["json_key"]
        hypothesis_key = metric_conf["hypothesis_key"]
        is_list = metric_conf["is_list_metric"]
        y_scale = metric_conf["y_scale"]
        y_label = metric_conf["y_label"]

        fixed_eps_plot1 = 0.5
        if fixed_eps_plot1 not in all_config_epsilons:
            print(
                f"Warning: Fixed epsilon {fixed_eps_plot1} for Plot 1 ({plot_name}) not in experiment epsilons. Skipping."
            )
        else:
            plot_performance_vs_alpha_modified(
                results_data=results_data,
                fixed_epsilon=fixed_eps_plot1,
                all_alphas=all_config_alphas,
                all_epsilons_config_list=all_config_epsilons,
                results_dir=results_dir,
                plot_type="all_algorithms",
                metric_json_key=json_key,
                data_hypothesis_key=hypothesis_key,
                is_list_metric=is_list,
                plot_y_label=y_label,
                plot_y_scale=y_scale,
                plot_filename_id=plot_id,
            )

        fixed_alpha_plot2 = 0.05
        if fixed_alpha_plot2 not in all_config_alphas:
            print(
                f"Warning: Fixed alpha {fixed_alpha_plot2} for Plot 2 ({plot_name}) not in experiment alphas. Skipping."
            )
        else:
            plot_performance_vs_epsilon_modified(
                results_data=results_data,
                fixed_alpha=fixed_alpha_plot2,
                all_epsilons=all_config_epsilons,
                all_epsilons_config_list=all_config_epsilons,
                results_dir=results_dir,
                plot_type="all_algorithms",
                metric_json_key=json_key,
                data_hypothesis_key=hypothesis_key,
                is_list_metric=is_list,
                plot_y_label=y_label,
                plot_y_scale=y_scale,
                plot_filename_id=plot_id,
            )

        dpsprt_alphas_plot3 = all_config_alphas
        plot_performance_vs_epsilon_modified(
            results_data=results_data,
            fixed_alpha=None,
            all_epsilons=all_config_epsilons,
            all_epsilons_config_list=all_config_epsilons,
            results_dir=results_dir,
            plot_type="dpsprt_multiple_alphas",
            dpsprt_alphas_for_plot=dpsprt_alphas_plot3,
            metric_json_key=json_key,
            data_hypothesis_key=hypothesis_key,
            is_list_metric=is_list,
            plot_y_label=y_label,
            plot_y_scale=y_scale,
            plot_filename_id=plot_id,
        )

        dpsprt_epsilons_plot4 = [
            e for e in [0.1, 0.5, 1.0, 5.0, 25.0, 100.0] if e in all_config_epsilons
        ]
        if not dpsprt_epsilons_plot4:
            dpsprt_epsilons_plot4 = all_config_epsilons

        plot_performance_vs_alpha_modified(
            results_data=results_data,
            fixed_epsilon=None,
            all_alphas=all_config_alphas,
            all_epsilons_config_list=all_config_epsilons,
            results_dir=results_dir,
            plot_type="dpsprt_multiple_epsilons",
            dpsprt_epsilons_for_plot=dpsprt_epsilons_plot4,
            metric_json_key=json_key,
            data_hypothesis_key=hypothesis_key,
            is_list_metric=is_list,
            plot_y_label=y_label,
            plot_y_scale=y_scale,
            plot_filename_id=plot_id,
        )

    print("\\n📊 All new performance plots generated.")


def plot_performance_vs_alpha_modified(
    results_data: Dict[str, Dict[str, Any]],
    all_alphas: List[float],
    all_epsilons_config_list: List[float],
    results_dir: str,
    plot_type: str,
    metric_json_key: str,
    data_hypothesis_key: str,
    is_list_metric: bool,
    plot_y_label: str,
    plot_y_scale: str,
    plot_filename_id: str,
    fixed_epsilon: Optional[float] = None,
    dpsprt_epsilons_for_plot: Optional[List[float]] = None,
) -> None:
    plots_dir = Path(results_dir) / "plots" / "performance"
    ensure_dir_exists(plots_dir)
    plt.figure(figsize=(5, 3.5))

    algorithm_styles = {
        "ClassicalSPRT": {"color": "#1f77b4", "label": "Classical SPRT", "marker": "o"},
        "DP-SPRT": {"color": "#ff7f0e", "label": "DP-SPRT Laplace", "marker": "s"},
        # "TunedDP-SPRT": {"color": "#2ca02c", "label": "Tuned DP-SPRT", "marker": "^"},
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

    title_metric_name = plot_y_label
    if plot_type == "all_algorithms":
        if fixed_epsilon is None:
            print(
                f"Warning: fixed_epsilon is None for plot_type 'all_algorithms' in plot_performance_vs_alpha_modified. Skipping plot for {plot_filename_id}."
            )
            return
        for algo_name, style in algorithm_styles.items():
            if algo_name in results_data:
                metric_means = []
                metric_stds: List[Optional[float]] = []
                valid_alphas_for_plot = []
                for alpha_v in sorted(all_alphas):
                    val_data = _get_metric_value(
                        results_data,
                        algo_name,
                        alpha_v,
                        fixed_epsilon,
                        all_epsilons_config_list,
                        metric_json_key,
                        data_hypothesis_key,
                        is_list_metric,
                    )
                    if val_data is not None:
                        if is_list_metric:
                            assert isinstance(
                                val_data, tuple
                            ), "Expected (mean, lower, upper) for list metric"
                            mean_val, lower_percentile, upper_percentile = val_data
                            metric_means.append(mean_val)
                            metric_stds.append((lower_percentile, upper_percentile))
                        else:
                            assert isinstance(
                                val_data, float
                            ), "Expected float for non-list metric"
                            metric_means.append(val_data)
                        valid_alphas_for_plot.append(alpha_v)

                if valid_alphas_for_plot:
                    if is_list_metric:
                        plt.errorbar(
                            valid_alphas_for_plot,
                            metric_means,
                            yerr=[
                                np.maximum(
                                    0,
                                    np.array(metric_means)
                                    - np.array([x[0] for x in metric_stds]),
                                ),
                                np.maximum(
                                    0,
                                    np.array([x[1] for x in metric_stds])
                                    - np.array(metric_means),
                                ),
                            ],
                            label=style["label"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                            capsize=3,
                            elinewidth=1,
                        )
                    else:
                        plt.plot(
                            valid_alphas_for_plot,
                            metric_means,
                            label=style["label"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                        )
        plt.title(f"{title_metric_name} vs. Alpha (ε = {fixed_epsilon})", fontsize=12)
        plot_filename = f"{plot_filename_id}_vs_alpha_fixed_eps_{fixed_epsilon}.pdf"

    elif plot_type == "dpsprt_multiple_epsilons":
        algo_name = "DP-SPRT"
        if algo_name in results_data and dpsprt_epsilons_for_plot:
            colors = (
                plt.cm.viridis(np.linspace(0, 1, len(dpsprt_epsilons_for_plot)))
                if len(dpsprt_epsilons_for_plot) > 5
                else None
            )
            for i, epsilon_v in enumerate(sorted(dpsprt_epsilons_for_plot)):
                metric_means = []
                metric_stds: List[Optional[Tuple[float, float]]] = []
                valid_alphas_for_plot = []
                for alpha_v in sorted(all_alphas):
                    val_data = _get_metric_value(
                        results_data,
                        algo_name,
                        alpha_v,
                        epsilon_v,
                        all_epsilons_config_list,
                        metric_json_key,
                        data_hypothesis_key,
                        is_list_metric,
                    )
                    if val_data is not None:
                        if is_list_metric:
                            assert isinstance(
                                val_data, tuple
                            ), "Expected (mean, lower, upper) for list metric"
                            mean_val, lower_percentile, upper_percentile = val_data
                            metric_means.append(mean_val)
                            metric_stds.append((lower_percentile, upper_percentile))
                        else:
                            assert isinstance(
                                val_data, float
                            ), "Expected float for non-list metric"
                            metric_means.append(val_data)
                        valid_alphas_for_plot.append(alpha_v)

                if valid_alphas_for_plot:
                    style = algorithm_styles.get(algo_name, {})
                    plot_label = f"{style.get('label', algo_name)} (ε={epsilon_v})"
                    plot_color = colors[i] if colors is not None else style.get("color")
                    if is_list_metric:
                        plt.errorbar(
                            valid_alphas_for_plot,
                            metric_means,
                            yerr=[
                                np.maximum(
                                    0,
                                    np.array(metric_means)
                                    - np.array([x[0] for x in metric_stds]),
                                ),
                                np.maximum(
                                    0,
                                    np.array([x[1] for x in metric_stds])
                                    - np.array(metric_means),
                                ),
                            ],
                            label=plot_label,
                            marker=style.get("marker", "s"),
                            color=plot_color,
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                            capsize=3,
                            elinewidth=1,
                        )
                    else:
                        plt.plot(
                            valid_alphas_for_plot,
                            metric_means,
                            label=plot_label,
                            marker=style.get("marker", "s"),
                            color=plot_color,
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                        )
        plt.title(f"DP-SPRT {title_metric_name} vs. Alpha (Multiple ε)", fontsize=12)
        plot_filename = f"dpsprt_{plot_filename_id}_vs_alpha_multi_eps.pdf"
    else:
        print(
            f"Warning: Unknown plot_type '{plot_type}' in plot_performance_vs_alpha_modified. Skipping plot for {plot_filename_id}."
        )
        return

    plt.xlabel("Alpha (α)", fontsize=11)
    plt.ylabel(plot_y_label, fontsize=11)
    plt.xscale("log")
    plt.yscale(plot_y_scale)
    plt.legend(fontsize=10)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.tight_layout(pad=0.5)
    save_path = plots_dir / plot_filename
    plt.savefig(save_path, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {save_path}")


def plot_performance_vs_epsilon_modified(
    results_data: Dict[str, Dict[str, Any]],
    all_epsilons: List[float],
    all_epsilons_config_list: List[float],
    results_dir: str,
    plot_type: str,
    metric_json_key: str,
    data_hypothesis_key: str,
    is_list_metric: bool,
    plot_y_label: str,
    plot_y_scale: str,
    plot_filename_id: str,
    fixed_alpha: Optional[float] = None,
    dpsprt_alphas_for_plot: Optional[List[float]] = None,
) -> None:
    plots_dir = Path(results_dir) / "plots" / "performance"
    ensure_dir_exists(plots_dir)
    plt.figure(figsize=(5, 3.5))

    algorithm_styles = {
        "ClassicalSPRT": {"color": "#1f77b4", "label": "Classical SPRT", "marker": "o"},
        "DP-SPRT": {"color": "#ff7f0e", "label": "DP-SPRT Laplace", "marker": "s"},
        # "TunedDP-SPRT": {"color": "#2ca02c", "label": "Tuned DP-SPRT", "marker": "^"},
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

    title_metric_name = plot_y_label
    if plot_type == "all_algorithms":
        if fixed_alpha is None:
            print(
                f"Warning: fixed_alpha is None for plot_type 'all_algorithms' in plot_performance_vs_epsilon_modified. Skipping plot for {plot_filename_id}."
            )
            return
        for algo_name, style in algorithm_styles.items():
            if algo_name in results_data:
                metric_means = []
                metric_stds: List[Optional[Tuple[float, float]]] = []
                valid_epsilons_for_plot = []
                for epsilon_v in sorted(all_epsilons):
                    if epsilon_v not in all_epsilons_config_list:
                        continue
                    val_data = _get_metric_value(
                        results_data,
                        algo_name,
                        fixed_alpha,
                        epsilon_v,
                        all_epsilons_config_list,
                        metric_json_key,
                        data_hypothesis_key,
                        is_list_metric,
                    )
                    if val_data is not None:
                        if is_list_metric:
                            assert isinstance(
                                val_data, tuple
                            ), "Expected (mean, lower, upper) for list metric"
                            mean_val, lower_percentile, upper_percentile = val_data
                            metric_means.append(mean_val)
                            metric_stds.append((lower_percentile, upper_percentile))
                        else:
                            assert isinstance(
                                val_data, float
                            ), "Expected float for non-list metric"
                            metric_means.append(val_data)
                        valid_epsilons_for_plot.append(epsilon_v)

                if valid_epsilons_for_plot:
                    if is_list_metric:
                        plt.errorbar(
                            valid_epsilons_for_plot,
                            metric_means,
                            yerr=[
                                np.maximum(
                                    0,
                                    np.array(metric_means)
                                    - np.array([x[0] for x in metric_stds]),
                                ),
                                np.maximum(
                                    0,
                                    np.array([x[1] for x in metric_stds])
                                    - np.array(metric_means),
                                ),
                            ],
                            label=style["label"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                            capsize=3,
                            elinewidth=1,
                        )
                    else:
                        plt.plot(
                            valid_epsilons_for_plot,
                            metric_means,
                            label=style["label"],
                            color=style["color"],
                            marker=style["marker"],
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                        )
        plt.title(f"{title_metric_name} vs. Epsilon", fontsize=12)
        plot_filename = f"{plot_filename_id}_vs_eps_fixed_alpha_{fixed_alpha}.pdf"

    elif plot_type == "dpsprt_multiple_alphas":
        algo_name = "DP-SPRT"
        if algo_name in results_data and dpsprt_alphas_for_plot:
            colors = (
                plt.cm.viridis(np.linspace(0, 1, len(dpsprt_alphas_for_plot)))
                if len(dpsprt_alphas_for_plot) > 5
                else None
            )
            for i, alpha_v in enumerate(sorted(dpsprt_alphas_for_plot)):
                metric_means = []
                metric_stds: List[Optional[Tuple[float, float]]] = []
                valid_epsilons_for_plot = []
                for epsilon_v in sorted(all_epsilons):
                    if epsilon_v not in all_epsilons_config_list:
                        continue
                    val_data = _get_metric_value(
                        results_data,
                        algo_name,
                        alpha_v,
                        epsilon_v,
                        all_epsilons_config_list,
                        metric_json_key,
                        data_hypothesis_key,
                        is_list_metric,
                    )
                    if val_data is not None:
                        if is_list_metric:
                            assert isinstance(
                                val_data, tuple
                            ), "Expected (mean, lower, upper) for list metric"
                            mean_val, lower_percentile, upper_percentile = val_data
                            metric_means.append(mean_val)
                            metric_stds.append((lower_percentile, upper_percentile))
                        else:
                            assert isinstance(
                                val_data, float
                            ), "Expected float for non-list metric"
                            metric_means.append(val_data)
                        valid_epsilons_for_plot.append(epsilon_v)

                if valid_epsilons_for_plot:
                    style = algorithm_styles.get(algo_name, {})
                    plot_label = f"{style.get('label', algo_name)} (α={alpha_v})"
                    plot_color = colors[i] if colors is not None else style.get("color")
                    if is_list_metric:
                        plt.errorbar(
                            valid_epsilons_for_plot,
                            metric_means,
                            yerr=[
                                np.maximum(
                                    0,
                                    np.array(metric_means)
                                    - np.array([x[0] for x in metric_stds]),
                                ),
                                np.maximum(
                                    0,
                                    np.array([x[1] for x in metric_stds])
                                    - np.array(metric_means),
                                ),
                            ],
                            label=plot_label,
                            marker=style.get("marker", "s"),
                            color=plot_color,
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                            capsize=3,
                            elinewidth=1,
                        )
                    else:
                        plt.plot(
                            valid_epsilons_for_plot,
                            metric_means,
                            label=plot_label,
                            marker=style.get("marker", "s"),
                            color=plot_color,
                            linestyle="-",
                            linewidth=1.5,
                            markersize=4,
                        )
        plt.title(f"DP-SPRT {title_metric_name} vs. Epsilon (Multiple α)", fontsize=12)
        plot_filename = f"dpsprt_{plot_filename_id}_vs_eps_multi_alpha.pdf"
    else:
        print(
            f"Warning: Unknown plot_type '{plot_type}' in plot_performance_vs_epsilon_modified. Skipping plot for {plot_filename_id}."
        )
        return

    if plot_type == "all_algorithms" and fixed_alpha is not None:
        if plot_filename_id == "type1_error":
            ref = plt.axhline(
                y=fixed_alpha,
                color=algorithm_styles["DP-SPRT"]["color"],
                linestyle=":",
                linewidth=1.2,
                label=f"Target α = {fixed_alpha}",
            )
        elif plot_filename_id == "type2_error":
            ref = plt.axhline(
                y=fixed_alpha,
                color=algorithm_styles["DP-SPRT"]["color"],
                linestyle=":",
                linewidth=1.2,
                label=f"Reference level (α = {fixed_alpha})",
            )

    plt.xlabel("Epsilon (ε)", fontsize=11)
    plt.ylabel(plot_y_label, fontsize=11)
    plt.xscale("log")
    plt.yscale(plot_y_scale)
    if plot_type == "all_algorithms" and (
        plot_filename_id == "type1_error" or plot_filename_id == "type2_error"
    ):
        plt.legend(handles=[ref], fontsize=10, loc="upper right")
    else:
        plt.legend(fontsize=10)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.tight_layout(pad=0.5)
    save_path = plots_dir / plot_filename
    plt.savefig(save_path, dpi=300, format="pdf")
    plt.close()
    print(f"Saved: {save_path}")


def load_and_plot_all_performance_metrics(
    results_file: str, results_dir: str = "results"
) -> None:
    """Load results data and config from a JSON file and generate all performance plots."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON from results file: {results_file}")
        return

    actual_results_data = data.get("results_data")
    if actual_results_data is None:
        actual_results_data = data.get("results")

    actual_config = data.get("config")
    if actual_config is None or not actual_config:
        actual_config = data.get("parameters")
        if actual_config:
            print(
                "ℹ️ Using 'parameters' key from JSON as configuration for performance plots."
            )
        else:
            actual_config = {}  # Default to empty dict if no config found

    if not actual_results_data:
        print(
            "❌ Neither 'results_data' nor 'results' key found in JSON file. Cannot generate performance plots."
        )
        return

    if not isinstance(actual_config, dict):
        print(
            f"❌ Expected 'config' or 'parameters' to be a dictionary, but got {type(actual_config)}. Performance plots may fail."
        )
        actual_config = {}

    print("\\n📊 Generating new performance plots...")
    plot_all_new_performance_plots(actual_results_data, actual_config, results_dir)
    # The original print statement from plot_all_new_performance_plots is removed from here
    # as it's now part of that function. If it was meant to be an overall completion message,
    # it can be added after this call.
    print("✅ New performance plots generated.")


if __name__ == "__main__":
    import argparse
    import json  # Added import for json

    parser = argparse.ArgumentParser(
        description="Generate performance metrics plots from SPRT experiment results."
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
        default="results",  # Consistent with plot_distributions.py
        help="Directory to save plots",
    )

    args = parser.parse_args()

    load_and_plot_all_performance_metrics(args.results_file, args.results_dir)
