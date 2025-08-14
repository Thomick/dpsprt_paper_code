"""
Setup and run script for SPRT comparison experiments.

This script provides a convenient way to install dependencies and run the
same experiments as in the original sprt_comparison.py script.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Define problem instances
PROBLEM_INSTANCES = [
    {"name": "easy_p0.3_p1.7", "mu0": "0.3", "mu1": "0.7"},
    {"name": "difficult_p0.45_p1.55", "mu0": "0.45", "mu1": "0.55"},
    {"name": "difficult_p0.05_p1.25", "mu0": "0.05", "mu1": "0.25"},
]


def run_experiments(fast_mode=False):
    """Run the experiments with the same parameters as the original script"""
    # Create the results directory
    results_dir = Path("results/sprt_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build the command
    cmd = [
        sys.executable,
        "main.py",
        "--mu0",
        "0.3",
        "--mu1",
        "0.7",
        "--epsilons",
        "0.1",
        "0.5",
        "1.0",
        "2.0",
        "5.0",
        "10.0",
        "25.0",
        "50.0",
        "100.0",
        "--alphas",
        "0.01",
        "0.025",
        "0.05",
        "0.075",
        "0.1",
        "0.15",
        "0.2",
        "--delta",
        "1e-5",
        "--A",
        "1.0",
        "--results-dir",
        str(results_dir),
        "--seed",
        "42",
    ]

    # Use reduced parameters if fast mode is enabled
    if fast_mode:
        cmd.extend(["--num-trials", "100", "--max-samples", "5000"])
        print("🚀 Running in fast mode with reduced parameters")
    else:
        cmd.extend(["--num-trials", "1000", "--max-samples", "50000"])
        print("🚀 Running with original parameters")

    # Run the experiments
    print("📊 Running SPRT comparison experiments...")
    try:
        subprocess.check_call(cmd)
        print("✅ Experiments completed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to run experiments")
        sys.exit(1)

    # Generate all plots by default
    generate_all_plots("results", "sprt_comparison")
    print(f"\n🎉 All done! Results and plots saved to {results_dir}")


def run_instance_comparison_experiments(fast_mode=False):
    """Run experiments for different problem instances."""
    base_results_dir_instances = Path("results/problem_instances")
    base_results_dir_instances.mkdir(parents=True, exist_ok=True)
    print(
        f"🗂️ Base results directory for instance comparison: {base_results_dir_instances}"
    )

    # Parameters that are common across instances (similar to run_experiments)
    common_epsilons = [
        "0.1",
        "0.5",
        "1.0",
        "2.0",
        "5.0",
        "10.0",
        "25.0",
        "50.0",
        "100.0",
    ]
    common_alphas = ["0.01", "0.025", "0.05", "0.075", "0.1", "0.15", "0.2"]
    common_delta = "1e-5"
    common_A = "1.0"
    common_seed = "42"

    for instance in PROBLEM_INSTANCES:
        instance_name = instance["name"]
        mu0 = instance["mu0"]
        mu1 = instance["mu1"]

        print(f"\n🧪 Processing instance: {instance_name} (mu0={mu0}, mu1={mu1})")

        instance_results_dir = base_results_dir_instances / instance_name
        instance_results_dir.mkdir(parents=True, exist_ok=True)

        cmd_main = [
            sys.executable,
            "main.py",
            "--mu0",
            mu0,
            "--mu1",
            mu1,
            "--epsilons",
            *common_epsilons,
            "--alphas",
            *common_alphas,
            "--delta",
            common_delta,
            "--A",
            common_A,
            "--results-dir",
            str(instance_results_dir),
            "--seed",
            common_seed,
        ]

        if fast_mode:
            cmd_main.extend(["--num-trials", "100", "--max-samples", "5000"])
            print(f"🚀 Running instance {instance_name} in fast mode.")
        else:
            cmd_main.extend(["--num-trials", "1000", "--max-samples", "10000"])
            print(f"🚀 Running instance {instance_name} with original parameters.")

        # Run main.py for the current instance
        print(f"📊 Running main.py for instance {instance_name}...")
        try:
            subprocess.check_call(cmd_main)
            print(f"✅ main.py completed successfully for instance {instance_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to run main.py for instance {instance_name}")
            # Optionally continue to next instance or exit
            # For now, let's print error and continue
            continue

        instance_results_file = instance_results_dir / "sprt_comparison_results.json"
        if not instance_results_file.exists():
            print(
                f"⚠️ Results file {instance_results_file} not found for instance {instance_name}. Skipping plot generation."
            )
            continue

        # Generate plots for this instance
        generate_distribution_plots(
            str(instance_results_file), str(instance_results_dir)
        )
        generate_performance_plots(
            str(instance_results_file), str(instance_results_dir)
        )

    print(
        f"\n🎉 All instance comparison experiments done! Results and plots saved under {base_results_dir_instances}"
    )

    # Call the new cross-instance plotting script
    print("📊 Generating DP-SPRT comparison plots across instances...")
    cross_instance_plots_dir = base_results_dir_instances / "cross_instance_plots"
    try:
        subprocess.check_call(
            [
                sys.executable,
                "plot_dp_sprt_across_instances.py",
                "--base-instance-results-dir",
                str(base_results_dir_instances),
                "--output-plots-dir",
                str(cross_instance_plots_dir),
                # Optionally, you can pass --fixed-alpha and --fixed-epsilon if you want non-default values
                # "--fixed-alpha", "0.05",
                # "--fixed-epsilon", "1.0",
            ]
        )
        print(
            f"✅ DP-SPRT cross-instance plots generated successfully in {cross_instance_plots_dir}"
        )
    except subprocess.CalledProcessError:
        print("❌ Failed to generate DP-SPRT cross-instance plots")
    except FileNotFoundError:  # If the new script is not found
        print(
            "❌ plot_dp_sprt_across_instances.py not found. Skipping cross-instance plot generation."
        )

    # Generate box plots after instance experiments
    generate_boxplots(
        main_results_dir="results", instance_results_dir=str(base_results_dir_instances)
    )


def generate_distribution_plots(results_file, results_dir):
    """Generate stopping time distribution plots for a results file."""
    print("📊 Generating stopping time distribution plots...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "plot_distributions.py",
                "--results-file",
                results_file,
                "--results-dir",
                results_dir,
            ]
        )
        print("✅ Distribution plots generated successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to generate distribution plots")
    except FileNotFoundError:
        print("❌ plot_distributions.py not found. Ensure the script exists.")


def generate_performance_plots(results_file, results_dir):
    """Generate performance metric plots for a results file."""
    print("📊 Generating performance metric plots...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "plot_performance_metrics.py",
                "--results-file",
                results_file,
                "--results-dir",
                results_dir,
            ]
        )
        print("✅ Performance metric plots generated successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to generate performance metric plots")
    except FileNotFoundError:
        print("❌ plot_performance_metrics.py not found. Ensure the script exists.")


def generate_boxplots(
    main_results_dir="results",
    instance_results_dir="results/problem_instances",
    fixed_alpha=0.05,
    fixed_epsilon=1.0,
):
    """Generate box plots comparing algorithms across instances, alphas, and epsilons."""
    print("📊 Generating box plots for SPRT comparison experiments...")

    try:
        subprocess.check_call(
            [
                sys.executable,
                "plot_boxplots.py",
                "--main-results-dir",
                main_results_dir,
                "--instance-results-dir",
                instance_results_dir,
                "--fixed-alpha",
                str(fixed_alpha),
                "--fixed-epsilon",
                str(fixed_epsilon),
            ]
        )
        print("✅ Box plots generated successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to generate box plots")
    except FileNotFoundError:
        print("❌ plot_boxplots.py not found. Ensure the script exists.")


def generate_all_plots(
    main_results_dir="results",
    main_experiment_subdir="sprt_comparison",
    fixed_alpha=0.05,
    fixed_epsilon=1.0,
):
    """Regenerate all plots from existing experiment results without rerunning experiments."""
    print("\n📊 Regenerating all plots from existing results...")

    # Main experiment directory
    main_exp_dir = Path(main_results_dir) / main_experiment_subdir
    main_results_file = main_exp_dir / "sprt_comparison_results.json"

    # Instance experiments directory
    instance_results_dir = Path(main_results_dir) / "problem_instances"

    # Check if main results exist
    if main_results_file.exists():
        print(f"📊 Generating plots for main experiment results: {main_results_file}")
        generate_distribution_plots(str(main_results_file), str(main_exp_dir))
        generate_performance_plots(str(main_results_file), str(main_results_dir))
    else:
        print(f"⚠️ Main results file not found: {main_results_file}")

    # Check if instance results exist
    if instance_results_dir.exists():
        print(f"📊 Generating plots for instance experiments: {instance_results_dir}")

        # Generate plots for each instance
        for instance in PROBLEM_INSTANCES:
            instance_name = instance["name"]
            instance_dir = instance_results_dir / instance_name
            instance_results_file = instance_dir / "sprt_comparison_results.json"

            if instance_results_file.exists():
                print(f"📊 Generating plots for instance: {instance_name}")
                generate_distribution_plots(
                    str(instance_results_file), str(instance_dir)
                )
                generate_performance_plots(
                    str(instance_results_file), str(instance_dir)
                )
            else:
                print(f"⚠️ Results file not found for instance: {instance_name}")

        # Generate cross-instance plots
        cross_instance_plots_dir = instance_results_dir / "cross_instance_plots"
        print("📊 Generating DP-SPRT comparison plots across instances...")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "plot_dp_sprt_across_instances.py",
                    "--base-instance-results-dir",
                    str(instance_results_dir),
                    "--output-plots-dir",
                    str(cross_instance_plots_dir),
                ]
            )
            print(f"✅ DP-SPRT cross-instance plots generated successfully")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ Failed to generate DP-SPRT cross-instance plots")
    else:
        print(f"⚠️ Instance results directory not found: {instance_results_dir}")

    # Generate box plots
    generate_boxplots(
        main_results_dir=main_results_dir,
        instance_results_dir=str(instance_results_dir),
        fixed_alpha=fixed_alpha,
        fixed_epsilon=fixed_epsilon,
    )

    print("✅ All plots regenerated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup and run SPRT comparison experiments"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run with reduced parameters for faster execution",
    )
    parser.add_argument(
        "--run-instance-comparison",
        action="store_true",
        help="Run experiments comparing different problem instances",
    )
    parser.add_argument(
        "--run-single-comparison",
        action="store_true",
        help="Run the original single set of SPRT comparison experiments (not instance-based)",
    )
    parser.add_argument(
        "--generate-boxplots",
        action="store_true",
        help="Generate box plots from existing results without running experiments",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate all plots from existing results without running experiments",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=0.05,
        help="Alpha value to use for fixed-alpha comparisons in box plots",
    )
    parser.add_argument(
        "--fixed-epsilon",
        type=float,
        default=1.0,
        help="Epsilon value to use for fixed-epsilon comparisons in box plots",
    )

    args = parser.parse_args()

    if args.plots_only:
        # Regenerate all plots without running experiments
        print("📊 Regenerating all plots without running experiments...")
        generate_all_plots(
            fixed_alpha=args.fixed_alpha,
            fixed_epsilon=args.fixed_epsilon,
        )
    elif args.generate_boxplots:
        print("📊 Generating box plots from existing results...")
        generate_boxplots(
            fixed_alpha=args.fixed_alpha,
            fixed_epsilon=args.fixed_epsilon,
        )
    elif args.run_single_comparison:
        print("🚀 Running single original SPRT comparison experiment...")
        run_experiments(fast_mode=args.fast)
    elif args.run_instance_comparison:  # Explicit request
        print("🚀 Running instance comparison experiments...")
        run_instance_comparison_experiments(fast_mode=args.fast)
    else:  # Default action: run instance comparison
        # This covers calls like:
        # python run_all.py
        # python run_all.py --fast
        # python run_all.py --skip-install (after install)
        # (and implicitly if --run-instance-comparison was the only flag, though caught above)
        print("🚀 Running instance comparison experiments (default)...")
        run_instance_comparison_experiments(fast_mode=args.fast)

    print("\n🏁 Script finished.")
