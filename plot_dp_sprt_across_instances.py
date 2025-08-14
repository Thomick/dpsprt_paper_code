import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

# Assuming utils.py is in the same directory or accessible via PYTHONPATH
# If utils.py is in the parent directory:
# import sys
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import (
    ensure_dir_exists,
)  # Make sure this import works based on your project structure

# Define problem instances (consistent with run_all.py)
PROBLEM_INSTANCES_CONFIG = [
    {
        "name": "easy_p0.3_p1.7",
        "display_name": "Easy (p₀=0.3, p₁=0.7)",
        "mu0": 0.3,
        "mu1": 0.7,
    },
    {
        "name": "difficult_p0.45_p1.55",
        "display_name": "Difficult (p₀=0.45, p₁=0.55)",
        "mu0": 0.45,
        "mu1": 0.55,
    },
    {
        "name": "difficult_p0.05_p1.25",
        "display_name": "Difficult (p₀=0.05, p₁=0.25)",
        "mu0": 0.05,
        "mu1": 0.25,
    },
]


# Helper function from plot_performance_metrics.py, adapted slightly
def _get_metric_value_for_instance(
    instance_results_data: Dict[str, Dict[str, Any]],
    algo_name: str,
    alpha_val: float,
    epsilon_val: float,
    all_epsilons_config_list: List[float],
    metric_json_key: str,
    data_hypothesis_key: str,
    is_list_metric: bool,
) -> Optional[float]:
    """Helper function to extract a metric value from a single instance's results data."""
    try:
        if str(alpha_val) not in instance_results_data.get(algo_name, {}):
            return None
        results_for_alpha = instance_results_data[algo_name][str(alpha_val)]

        epsilon_idx: Optional[int] = None
        try:
            epsilon_idx = all_epsilons_config_list.index(epsilon_val)
        except ValueError:
            # If epsilon_val is not in the list for this specific instance (e.g. different config)
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
                return np.mean(metric_val)
            return None
        else:
            try:
                return float(metric_val)
            except (ValueError, TypeError):
                return None
    except Exception:
        return None


def plot_dpsprt_comparison_across_instances(
    base_instance_results_dir: str,
    output_plots_dir: str,
    fixed_alpha_for_eps_plots: float = 0.05,
    fixed_epsilon_for_alpha_plots: float = 1.0,
):
    """
    Generates plots comparing DP-SPRT performance across different problem instances.
    """
    ensure_dir_exists(output_plots_dir)
    print(
        f"🔄 Loading results and generating cross-instance plots in {output_plots_dir}"
    )

    all_loaded_data = {}
    # First, try to get a common set of epsilons and alphas from the first successfully loaded config
    common_config_epsilons = None
    common_config_alphas = None

    for instance_conf in PROBLEM_INSTANCES_CONFIG:
        instance_name = instance_conf["name"]
        results_file_path = (
            Path(base_instance_results_dir)
            / instance_name
            / "sprt_comparison_results.json"
        )
        if results_file_path.exists():
            try:
                with open(results_file_path, "r") as f:
                    data = json.load(f)

                results = data.get("results_data") or data.get("results")
                config = data.get("config") or data.get("parameters", {})

                if results:
                    all_loaded_data[instance_name] = {
                        "results": results,
                        "config": config,
                        "display_name": instance_conf["display_name"],
                    }
                    if common_config_epsilons is None and "epsilons" in config:
                        common_config_epsilons = sorted(
                            [float(e) for e in config["epsilons"]]
                        )
                    if (
                        common_config_alphas is None and "alphas_betas" in config
                    ):  # or just "alphas"
                        common_config_alphas = sorted(
                            [float(a) for a in config["alphas_betas"]]
                        )
                    print(f"✅ Loaded results for instance: {instance_name}")
                else:
                    print(
                        f"⚠️ No 'results_data' or 'results' key in {results_file_path} for {instance_name}"
                    )
            except json.JSONDecodeError:
                print(
                    f"❌ Error decoding JSON from {results_file_path} for {instance_name}"
                )
            except Exception as e:
                print(
                    f"❌ Unexpected error loading {results_file_path} for {instance_name}: {e}"
                )
        else:
            print(
                f"⚠️ Results file not found for instance: {instance_name} at {results_file_path}"
            )

    if not all_loaded_data:
        print(
            "❌ No data loaded for any instance. Cannot generate cross-instance plots."
        )
        return

    # Fallback if config parsing failed
    if common_config_epsilons is None:
        common_config_epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        print(f"⚠️ Using fallback epsilons: {common_config_epsilons}")
    if common_config_alphas is None:
        common_config_alphas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
        print(f"⚠️ Using fallback alphas: {common_config_alphas}")

    metrics_to_plot_config = [
        {
            "id": "mst_h0",
            "name": "Mean Stopping Time (H0)",
            "json_key": "stopping_times",
            "hypothesis_key": "H0",
            "is_list": True,
            "y_scale": "log",
        },
        {
            "id": "mst_h1",
            "name": "Mean Stopping Time (H1)",
            "json_key": "stopping_times",
            "hypothesis_key": "H1",
            "is_list": True,
            "y_scale": "log",
        },
        {
            "id": "type1_error",
            "name": "Type 1 Error (Empirical α)",
            "json_key": "type_i_rate",
            "hypothesis_key": "metrics",
            "is_list": False,
            "y_scale": "linear",
        },
        {
            "id": "type2_error",
            "name": "Type 2 Error (Empirical β)",
            "json_key": "type_ii_rate",
            "hypothesis_key": "metrics",
            "is_list": False,
            "y_scale": "linear",
        },
    ]

    algo_to_plot = "DP-SPRT"  # Focus of this plot
    dpsprt_style = {"color": "#ff7f0e", "marker": "s"}
    dpsprt_gaussian_style = {"color": "#800080", "marker": "*"}
    classical_sprt_style = {
        "color": "#1f77b4",
        "linestyle": "--",
        "linewidth": 1.2,
        "marker": None,
    }

    # --- Plot 1: Metric vs. Epsilon for DP-SPRT (fixed alpha, multiple instances) ---
    if fixed_alpha_for_eps_plots not in common_config_alphas:
        print(
            f"⚠️ Fixed alpha {fixed_alpha_for_eps_plots} for (Metric vs Epsilon) plots not in common_config_alphas {common_config_alphas}. Using first available alpha: {common_config_alphas[0] if common_config_alphas else 'N/A'}"
        )
        if not common_config_alphas:
            print("❌ No alphas available. Skipping Metric vs Epsilon plots.")
            return  # Or skip this block
        fixed_alpha_for_eps_plots = common_config_alphas[0]

    for metric_conf in metrics_to_plot_config:
        plt.figure(figsize=(6, 4))
        plot_title = f"DP-SPRT: {metric_conf['name']} vs. Epsilon (α = {fixed_alpha_for_eps_plots})"
        plot_filename = f"dpsprt_{metric_conf['id']}_vs_eps_instances_alpha{fixed_alpha_for_eps_plots}.pdf"

        has_data_for_plot1 = False
        for instance_name, data_pack in all_loaded_data.items():
            instance_results = data_pack["results"]
            instance_display_name = data_pack["display_name"]
            # Use common_config_epsilons for iteration, but instance_results should contain data for these
            # The _get_metric_value_for_instance will use the config from THAT instance if needed, but here we align axes.

            # Ensure this instance's config epsilons are used by _get_metric_value_for_instance correctly
            # We primarily use common_config_epsilons for the X-axis points.
            instance_specific_epsilons_from_its_config = sorted(
                [
                    float(e)
                    for e in data_pack.get("config", {}).get(
                        "epsilons", common_config_epsilons
                    )
                ]
            )

            metric_values = []
            valid_epsilons_for_plot = []
            for (
                eps_val
            ) in (
                common_config_epsilons
            ):  # Iterate over a common set of epsilons for the X-axis
                val = _get_metric_value_for_instance(
                    instance_results,
                    algo_to_plot,
                    fixed_alpha_for_eps_plots,
                    eps_val,
                    instance_specific_epsilons_from_its_config,  # Pass the specific epsilons for this instance's data structure
                    metric_conf["json_key"],
                    metric_conf["hypothesis_key"],
                    metric_conf["is_list"],
                )
                if val is not None:
                    metric_values.append(val)
                    valid_epsilons_for_plot.append(eps_val)

            if valid_epsilons_for_plot:
                has_data_for_plot1 = True
                plt.plot(
                    valid_epsilons_for_plot,
                    metric_values,
                    label=f"{instance_display_name} (Laplace)",
                    marker=dpsprt_style["marker"],
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                )

            # Add DP-SPRT Gaussian
            metric_values_gaussian = []
            valid_epsilons_for_plot_gaussian = []
            for eps_val_g in common_config_epsilons:
                val_g = _get_metric_value_for_instance(
                    instance_results,
                    "DP-SPRT-Gaussian",
                    fixed_alpha_for_eps_plots,
                    eps_val_g,
                    instance_specific_epsilons_from_its_config,
                    metric_conf["json_key"],
                    metric_conf["hypothesis_key"],
                    metric_conf["is_list"],
                )
                if val_g is not None:
                    metric_values_gaussian.append(val_g)
                    valid_epsilons_for_plot_gaussian.append(eps_val_g)

            if valid_epsilons_for_plot_gaussian:
                has_data_for_plot1 = True
                plt.plot(
                    valid_epsilons_for_plot_gaussian,
                    metric_values_gaussian,
                    label=f"{instance_display_name} (Gaussian)",
                    marker=dpsprt_gaussian_style["marker"],
                    color=dpsprt_gaussian_style["color"],
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                )

        if not has_data_for_plot1:
            print(f"SKIPPING plot: {plot_title} - no data found for any instance.")
            plt.close()
            continue

        plt.xlabel("Epsilon (ε)", fontsize=9)
        plt.ylabel(metric_conf["name"], fontsize=9)
        plt.xscale("log")
        plt.yscale(metric_conf["y_scale"])

        # Add Classical SPRT reference lines for MST plots
        if metric_conf["id"] in ["mst_h0", "mst_h1"] and has_data_for_plot1:
            for instance_name_ref, data_pack_ref in all_loaded_data.items():
                instance_results_ref = data_pack_ref["results"]
                instance_display_name_ref = data_pack_ref["display_name"]
                instance_specific_epsilons_ref = sorted(
                    [
                        float(e)
                        for e in data_pack_ref.get("config", {}).get(
                            "epsilons", common_config_epsilons
                        )
                    ]
                )

                if (
                    not instance_specific_epsilons_ref
                ):  # Should not happen if data was plotted for DP-SPRT
                    continue

                # Epsilon value for Classical SPRT is a placeholder as it's epsilon-independent for fixed alpha
                classical_sprt_epsilon_placeholder = instance_specific_epsilons_ref[0]

                classical_mst_val = _get_metric_value_for_instance(
                    instance_results_ref,
                    "ClassicalSPRT",
                    fixed_alpha_for_eps_plots,
                    classical_sprt_epsilon_placeholder,
                    instance_specific_epsilons_ref,
                    metric_conf["json_key"],
                    metric_conf["hypothesis_key"],
                    metric_conf["is_list"],
                )

                if classical_mst_val is not None:
                    plt.axhline(
                        y=classical_mst_val,
                        color=classical_sprt_style["color"],
                        linestyle=classical_sprt_style["linestyle"],
                        linewidth=classical_sprt_style["linewidth"],
                        label=f"Classical SPRT Ref. ({instance_display_name_ref})",
                    )

        if metric_conf["id"] == "type1_error":
            plt.axhline(
                y=fixed_alpha_for_eps_plots,
                color=dpsprt_style["color"],
                linestyle="--",
                linewidth=1,
                label=f"Target α = {fixed_alpha_for_eps_plots}",
            )
        elif metric_conf["id"] == "type2_error":
            plt.axhline(
                y=fixed_alpha_for_eps_plots,
                color=dpsprt_style["color"],
                linestyle="--",
                linewidth=1,
                label=f"Target β = {fixed_alpha_for_eps_plots}",
            )

        plt.title(plot_title, fontsize=10)
        plt.legend(fontsize=8)
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout(pad=0.5)
        save_path = Path(output_plots_dir) / plot_filename
        plt.savefig(save_path, dpi=300, format="pdf")
        plt.close()
        print(f"Saved: {save_path}")

    # --- Plot 2: Metric vs. Alpha for DP-SPRT (fixed epsilon, multiple instances) ---
    if fixed_epsilon_for_alpha_plots not in common_config_epsilons:
        print(
            f"⚠️ Fixed epsilon {fixed_epsilon_for_alpha_plots} for (Metric vs Alpha) plots not in common_config_epsilons {common_config_epsilons}. Using first available epsilon: {common_config_epsilons[0] if common_config_epsilons else 'N/A'}"
        )
        if not common_config_epsilons:
            print("❌ No epsilons available. Skipping Metric vs Alpha plots.")
            return  # or skip this block
        fixed_epsilon_for_alpha_plots = common_config_epsilons[0]

    for metric_conf in metrics_to_plot_config:
        plt.figure(figsize=(6, 4))
        plot_title = f"DP-SPRT: {metric_conf['name']} vs. Alpha (ε = {fixed_epsilon_for_alpha_plots})"
        plot_filename = f"dpsprt_{metric_conf['id']}_vs_alpha_instances_eps{fixed_epsilon_for_alpha_plots}.pdf"

        has_data_for_plot2 = False
        for instance_name, data_pack in all_loaded_data.items():
            instance_results = data_pack["results"]
            instance_display_name = data_pack["display_name"]
            instance_specific_epsilons_from_its_config = sorted(
                [
                    float(e)
                    for e in data_pack.get("config", {}).get(
                        "epsilons", common_config_epsilons
                    )
                ]
            )

            metric_values = []
            valid_alphas_for_plot = []
            for alpha_val in common_config_alphas:  # Iterate common alphas for X-axis
                val = _get_metric_value_for_instance(
                    instance_results,
                    algo_to_plot,
                    alpha_val,
                    fixed_epsilon_for_alpha_plots,
                    instance_specific_epsilons_from_its_config,  # Pass the specific epsilons for this instance's data structure
                    metric_conf["json_key"],
                    metric_conf["hypothesis_key"],
                    metric_conf["is_list"],
                )
                if val is not None:
                    metric_values.append(val)
                    valid_alphas_for_plot.append(alpha_val)

            if valid_alphas_for_plot:
                has_data_for_plot2 = True
                plt.plot(
                    valid_alphas_for_plot,
                    metric_values,
                    label=f"{instance_display_name} (Laplace)",
                    marker=dpsprt_style["marker"],
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                )

            # Add DP-SPRT Gaussian
            metric_values_gaussian_alpha = []
            valid_alphas_for_plot_gaussian = []
            for alpha_val_g in common_config_alphas:
                val_g = _get_metric_value_for_instance(
                    instance_results,
                    "DP-SPRT-Gaussian",
                    alpha_val_g,
                    fixed_epsilon_for_alpha_plots,
                    instance_specific_epsilons_from_its_config,
                    metric_conf["json_key"],
                    metric_conf["hypothesis_key"],
                    metric_conf["is_list"],
                )
                if val_g is not None:
                    metric_values_gaussian_alpha.append(val_g)
                    valid_alphas_for_plot_gaussian.append(alpha_val_g)

            if valid_alphas_for_plot_gaussian:
                has_data_for_plot2 = True
                plt.plot(
                    valid_alphas_for_plot_gaussian,
                    metric_values_gaussian_alpha,
                    label=f"{instance_display_name} (Gaussian)",
                    marker=dpsprt_gaussian_style["marker"],
                    color=dpsprt_gaussian_style["color"],
                    linestyle="-",
                    linewidth=1.5,
                    markersize=5,
                )

        if not has_data_for_plot2:
            print(f"SKIPPING plot: {plot_title} - no data found for any instance.")
            plt.close()
            continue

        plt.xlabel("Alpha (α)", fontsize=9)
        plt.ylabel(metric_conf["name"], fontsize=9)
        plt.xscale("log")
        plt.yscale(metric_conf["y_scale"])

        # Add Classical SPRT reference lines for MST plots
        if metric_conf["id"] in ["mst_h0", "mst_h1"] and has_data_for_plot2:
            for instance_name_ref, data_pack_ref in all_loaded_data.items():
                instance_results_ref = data_pack_ref["results"]
                instance_display_name_ref = data_pack_ref["display_name"]
                instance_specific_epsilons_ref = sorted(
                    [
                        float(e)
                        for e in data_pack_ref.get("config", {}).get(
                            "epsilons", common_config_epsilons
                        )
                    ]
                )

                if (
                    not instance_specific_epsilons_ref
                ):  # Should not happen if data was plotted for DP-SPRT
                    continue

                classical_mst_values_for_alphas = []
                classical_valid_alphas_for_plot = []

                for alpha_val_ref in common_config_alphas:
                    # fixed_epsilon_for_alpha_plots is used as the epsilon placeholder for ClassicalSPRT
                    classical_mst_val = _get_metric_value_for_instance(
                        instance_results_ref,
                        "ClassicalSPRT",
                        alpha_val_ref,
                        fixed_epsilon_for_alpha_plots,
                        instance_specific_epsilons_ref,
                        metric_conf["json_key"],
                        metric_conf["hypothesis_key"],
                        metric_conf["is_list"],
                    )
                    if classical_mst_val is not None:
                        classical_mst_values_for_alphas.append(classical_mst_val)
                        classical_valid_alphas_for_plot.append(alpha_val_ref)

                if classical_valid_alphas_for_plot:
                    plt.plot(
                        classical_valid_alphas_for_plot,
                        classical_mst_values_for_alphas,
                        label=f"Classical SPRT Ref. ({instance_display_name_ref})",
                        color=classical_sprt_style["color"],
                        linestyle=classical_sprt_style["linestyle"],
                        linewidth=classical_sprt_style["linewidth"],
                        marker=classical_sprt_style["marker"],
                    )

        # For Type I error plots vs Alpha, the target alpha IS the x-axis, so a horizontal line based on a fixed alpha is not standard.
        # If we wanted to show a reference line for e.g. type_ii_error == alpha, that could be added.

        plt.title(plot_title, fontsize=10)
        plt.legend(fontsize=8)
        plt.tick_params(axis="both", which="major", labelsize=8)
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.tight_layout(pad=0.5)
        save_path = Path(output_plots_dir) / plot_filename
        plt.savefig(save_path, dpi=300, format="pdf")
        plt.close()
        print(f"Saved: {save_path}")

    print("✅ Cross-instance DP-SPRT comparison plots generated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate DP-SPRT performance comparison plots across different problem instances."
    )
    parser.add_argument(
        "--base-instance-results-dir",
        type=str,
        default="results/problem_instances",
        help="Base directory where subdirectories for each instance's results are located.",
    )
    parser.add_argument(
        "--output-plots-dir",
        type=str,
        default="results/problem_instances/cross_instance_plots",
        help="Directory to save the generated cross-instance plots.",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=0.05,
        help="Fixed alpha value for plots showing metric vs. epsilon.",
    )
    parser.add_argument(
        "--fixed-epsilon",
        type=float,
        default=1.0,
        help="Fixed epsilon value for plots showing metric vs. alpha.",
    )
    args = parser.parse_args()

    # Ensure the utils import works, this might need adjustment based on actual file structure
    # If plot_dp_sprt_across_instances.py is in the root alongside utils.py:
    # (no change needed to sys.path)
    # If plot_dp_sprt_across_instances.py is in a subdirectory e.g. 'plotting_scripts' and utils.py is in root:
    # import sys
    # sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add root to sys.path
    # from utils import ensure_dir_exists

    # For the current structure where utils.py is assumed to be at the same level or findable
    # (as per the original plot_performance_metrics.py)

    plot_dpsprt_comparison_across_instances(
        base_instance_results_dir=args.base_instance_results_dir,
        output_plots_dir=args.output_plots_dir,
        fixed_alpha_for_eps_plots=args.fixed_alpha,
        fixed_epsilon_for_alpha_plots=args.fixed_epsilon,
    )
