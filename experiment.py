"""
Functions and classes for running SPRT experiments.

This module provides the ExperimentRunner class for running experiments
with different SPRT algorithms and analyzing the results.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from datetime import datetime
import logging
from tqdm import tqdm

from config import ExperimentConfig
from sprt import (
    create_classical_sprt,
    create_dp_sprt,
    create_tuned_dp_sprt,
    create_priv_sprt,
    create_dp_sprt_subsampled,
    SPRT,
    ClassicalSPRT,
    DPSPRT,
    TunedDPSPRT,
    PrivSPRT,
    DPSPRTSubsampled,
    create_dp_sprt_gaussian,
    DPSPRTGaussian,
)


class ExperimentRunner:
    """Runner for SPRT experiments.

    This class manages the execution of experiments with different SPRT algorithms.
    It provides methods for generating data, running single and batch experiments,
    and analyzing the results.

    Attributes:
        verbose: Whether to print progress information
        use_tqdm: Whether to use tqdm progress bars
    """

    def __init__(self, verbose: bool = False, use_tqdm: bool = True):
        """Initialize the experiment runner.

        Args:
            verbose: Whether to print progress information
            use_tqdm: Whether to use tqdm progress bars
        """
        self.verbose = verbose
        self.use_tqdm = use_tqdm

        if verbose:
            # Set up console logger if not already configured
            if not logging.getLogger().handlers:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )

    def generate_batch_data(
        self, key: jnp.ndarray, mu: float, batch_size: int, max_samples: int
    ) -> jnp.ndarray:
        """Generate batches of Bernoulli sequences.

        Args:
            key: JAX random key
            mu: Probability parameter for Bernoulli distribution
            batch_size: Number of sequences to generate
            max_samples: Maximum length of each sequence

        Returns:
            Array of shape (batch_size, max_samples) containing Bernoulli samples
        """
        return random.bernoulli(key, p=mu, shape=(batch_size, max_samples))

    def create_algorithm_from_config(self, config: ExperimentConfig) -> SPRT:
        """Create an SPRT algorithm instance from a configuration.

        Args:
            config: Experiment configuration

        Returns:
            An SPRT algorithm instance

        Raises:
            ValueError: If the algorithm method is unknown
        """
        if config.method == "SPRT":
            return create_classical_sprt(
                config.mu0, config.mu1, config.alpha, config.beta
            )

        elif config.method == "DP-SPRT":
            if config.epsilon is None:
                raise ValueError("epsilon must be specified for DP-SPRT")
            return create_dp_sprt(
                config.mu0, config.mu1, config.alpha, config.beta, config.epsilon
            )

        elif config.method == "Tuned DP-SPRT":
            if config.epsilon is None:
                raise ValueError("epsilon must be specified for Tuned DP-SPRT")
            if config.c1 is None or config.c2 is None:
                raise ValueError("c1 and c2 must be specified for Tuned DP-SPRT")
            return create_tuned_dp_sprt(
                config.mu0,
                config.mu1,
                config.alpha,
                config.beta,
                config.epsilon,
                config.c1,
                config.c2,
            )
        elif config.method == "Priv-SPRT":
            if config.epsilon is None:
                raise ValueError("epsilon must be specified for Priv-SPRT")
            if config.delta is None:
                raise ValueError("delta must be specified for Priv-SPRT")
            if config.A is None:
                raise ValueError("A must be specified for Priv-SPRT")

            return create_priv_sprt(
                config.mu0,
                config.mu1,
                config.alpha,
                config.beta,
                config.epsilon,
                config.delta,
                config.A,
            )

        elif config.method == "DP-SPRT-Subsampled":
            if config.epsilon is None:
                raise ValueError("epsilon must be specified for DP-SPRT-Subsampled")
            return create_dp_sprt_subsampled(
                config.mu0, config.mu1, config.alpha, config.beta, config.epsilon
            )

        elif config.method == "DP-SPRT-Gaussian":
            if config.epsilon is None:
                raise ValueError("epsilon must be specified for DP-SPRT-Gaussian")
            if config.delta is None:
                raise ValueError("delta must be specified for DP-SPRT-Gaussian")
            return create_dp_sprt_gaussian(
                config.mu0,
                config.mu1,
                config.alpha,
                config.beta,
                config.epsilon,
                config.delta,
            )

        else:
            raise ValueError(f"Unknown method: {config.method}")

    def run_hypothesis_test(
        self,
        algorithm: SPRT,
        hypothesis: str,
        n_simulations: int,
        max_samples: int,
        seed: Optional[int] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a hypothesis test with the given algorithm.

        Args:
            algorithm: SPRT algorithm instance
            hypothesis: Either "H0" (null hypothesis) or "H1" (alternative hypothesis)
            n_simulations: Number of simulations to run
            max_samples: Maximum number of samples per sequence
            seed: Random seed for reproducibility
            return_details: Whether to return detailed results

        Returns:
            Dictionary containing test results

        Raises:
            ValueError: If the hypothesis is not "H0" or "H1"
        """
        if hypothesis not in ["H0", "H1"]:
            raise ValueError(f"hypothesis must be 'H0' or 'H1', got {hypothesis}")

        # Determine the true parameter value based on the hypothesis
        true_mu = algorithm.params.mu0 if hypothesis == "H0" else algorithm.params.mu1

        # Set up random key
        if seed is not None:
            key = random.PRNGKey(seed)
        else:
            key = random.PRNGKey(0)

        # Generate data
        key, subkey = random.split(key)
        data = self.generate_batch_data(subkey, true_mu, n_simulations, max_samples)

        # For DP variants, generate random keys for each simulation
        needs_keys = isinstance(
            algorithm, (DPSPRT, TunedDPSPRT, PrivSPRT, DPSPRTSubsampled, DPSPRTGaussian)
        )
        if needs_keys:
            key, subkey = random.split(key)
            keys = random.split(subkey, n_simulations)
        else:
            keys = None

        # Run batch test
        if self.verbose:
            print(f"Running {n_simulations} simulations for {hypothesis}...")

        results = algorithm.batch_test(data, keys, return_details)

        # Add metadata
        results["hypothesis"] = hypothesis
        results["true_mu"] = true_mu

        return results

    def run_single_experiment(
        self, config: ExperimentConfig, return_details: bool = False
    ) -> Dict[str, Any]:
        """Run a single experiment with the given configuration.

        Args:
            config: Experiment configuration
            return_details: Whether to return detailed results

        Returns:
            Dictionary containing experiment results, metrics, and configuration
        """
        # Validate configuration
        config.validate()

        # Create algorithm
        algorithm = self.create_algorithm_from_config(config)

        # Run H0 tests
        h0_results = self.run_hypothesis_test(
            algorithm,
            "H0",
            config.n_sims,
            config.max_samples,
            config.seed,
            return_details,
        )

        # Run H1 tests
        h1_seed = config.seed + 10000 if config.seed is not None else None
        h1_results = self.run_hypothesis_test(
            algorithm, "H1", config.n_sims, config.max_samples, h1_seed, return_details
        )

        # Calculate performance metrics
        metrics = {
            "type_i_rate": float(np.mean(h0_results["decisions"] == 1)),
            "type_ii_rate": float(np.mean(h1_results["decisions"] == -1)),
            "avg_sample_size": (
                float(np.mean(h0_results["stopping_times"]))
                + float(np.mean(h1_results["stopping_times"]))
            )
            / 2,
            "avg_samples_h0": float(np.mean(h0_results["stopping_times"])),
            "avg_samples_h1": float(np.mean(h1_results["stopping_times"])),
        }

        # Add PrivSPRT specific metrics if applicable
        if isinstance(algorithm, PrivSPRT):
            if algorithm.params.threshold_a is not None:
                metrics["threshold_a"] = float(algorithm.params.threshold_a)
            if algorithm.params.threshold_b is not None:
                metrics["threshold_b"] = float(algorithm.params.threshold_b)

        return {
            "config": config.to_dict(),
            "metrics": metrics,
            "H0": h0_results,
            "H1": h1_results,
        }

    def run_experiment_suite(
        self,
        configs: Dict[str, ExperimentConfig],
        return_details: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Run a suite of experiments.

        Args:
            configs: Dictionary mapping experiment names to their configurations
            return_details: Whether to return detailed results

        Returns:
            Dictionary mapping experiment names to their results
        """
        # Run experiments
        all_results = {}
        if self.verbose:
            print(f"Running {len(configs)} experiments...")

        # Use tqdm for progress tracking if enabled
        iterator = tqdm(configs.items()) if self.use_tqdm else configs.items()

        for name, config in iterator:
            if self.use_tqdm:
                iterator.set_description(f"Running {name}")
            elif self.verbose:
                print(f"Running experiment: {name}")

            result_data = self.run_single_experiment(config, return_details)
            all_results[name] = result_data

        return all_results


def run_algorithm_comparison(
    mu0: float,
    mu1: float,
    epsilons: list,
    alphas_betas: list,
    delta: float = 1e-5,
    A: float = 1.0,
    num_trials: int = 1000,
    max_samples: int = 10000,
    seed: int = 42,
    use_tqdm: bool = True,
    results_dir: str = "results",
) -> Dict[str, Dict[float, List[Dict[str, Any]]]]:
    """Run a comparison of different SPRT algorithms.

    Args:
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        epsilons: List of privacy parameters to test
        alphas_betas: List of alpha=beta values to test
        delta: Privacy failure probability for PrivSPRT
        A: Truncation parameter for PrivSPRT
        num_trials: Number of trials per configuration
        max_samples: Maximum number of samples per sequence
        seed: Random seed
        use_tqdm: Whether to use tqdm progress bars
        results_dir: Directory to save results

    Returns:
        Dictionary of results
    """
    from pathlib import Path
    from utils import ensure_dir_exists, save_json, save_results_to_dataframe

    start_time = datetime.now()

    # Initialize random key
    key = random.PRNGKey(seed)

    # Ensure results directory exists
    ensure_dir_exists(Path(results_dir))

    # Define algorithm names
    algorithm_names = [
        "ClassicalSPRT",
        "DP-SPRT",
        "TunedDP-SPRT",
        "Priv-SPRT",
        "DP-SPRT-Subsampled",
        "DP-SPRT-Gaussian",
    ]

    # Initialize results dictionary
    results_dict = {name: {ab: [] for ab in alphas_betas} for name in algorithm_names}

    # Create experiment runner
    runner = ExperimentRunner(verbose=True, use_tqdm=use_tqdm)

    # Run experiments for each alpha=beta value and epsilon
    for alpha_beta in alphas_betas:
        print(f"Processing alpha=beta={alpha_beta}")

        for epsilon in tqdm(epsilons, desc=f"Testing epsilons (α=β={alpha_beta})"):
            # Base experiment configuration
            base_config = {
                "mu0": mu0,
                "mu1": mu1,
                "alpha": alpha_beta,
                "beta": alpha_beta,
                "n_sims": num_trials,
                "max_samples": max_samples,
                "seed": seed,
            }

            # Create configs for standard algorithms
            variants = {
                "ClassicalSPRT": {"method": "SPRT"},
                "DP-SPRT": {"method": "DP-SPRT", "epsilon": epsilon},
                "TunedDP-SPRT": {
                    "method": "Tuned DP-SPRT",
                    "epsilon": epsilon,
                    "c1": 1.0,
                    "c2": 0.5,
                },
                "Priv-SPRT": {
                    "method": "Priv-SPRT",
                    "epsilon": epsilon,
                    "delta": delta,
                    "A": A,
                },
                "DP-SPRT-Subsampled": {
                    "method": "DP-SPRT-Subsampled",
                    "epsilon": epsilon,
                },
                "DP-SPRT-Gaussian": {
                    "method": "DP-SPRT-Gaussian",
                    "epsilon": epsilon,
                    "delta": delta,
                },
            }

            # Create experiment configs
            configs = {}
            for name, variant_params in variants.items():
                # Merge base config with variant-specific parameters
                config_dict = base_config.copy()
                config_dict.update(variant_params)
                # Create the configuration
                configs[name] = ExperimentConfig.from_dict(config_dict)

            # Run experiments
            all_results = runner.run_experiment_suite(configs)

            # Extract and store metrics from experiments
            for name in algorithm_names:
                if name in all_results:
                    # all_results[name] is the dictionary:
                    # { "config": ..., "metrics": ..., "H0": ..., "H1": ... }
                    # We append this entire dictionary.
                    results_dict[name][alpha_beta].append(all_results[name])
                else:
                    # Append placeholder if results for an algorithm are missing for this epsilon
                    results_dict[name][alpha_beta].append(
                        {
                            "config": {},  # Or relevant empty config
                            "metrics": {
                                "type_i_rate": np.nan,
                                "type_ii_rate": np.nan,
                                "avg_sample_size": np.nan,
                                "avg_samples_h0": np.nan,
                                "avg_samples_h1": np.nan,
                            },
                            "H0": {
                                "stopping_times": [],
                                "decisions": [],
                            },  # Empty lists
                            "H1": {
                                "stopping_times": [],
                                "decisions": [],
                            },  # Empty lists
                        }
                    )

    # Save results as JSON
    results_file = Path(results_dir) / "sprt_comparison_results.json"
    save_json(
        {
            "parameters": {
                "mu0": mu0,
                "mu1": mu1,
                "epsilons": epsilons,
                "alphas_betas": alphas_betas,
                "delta": delta,
                "A": A,
                "num_trials": num_trials,
                "max_samples": max_samples,
                "seed": seed,
            },
            "results": results_dict,
        },
        results_file,
    )

    # Save results as CSV
    csv_file = Path(results_dir) / "sprt_comparison_results.csv"
    save_results_to_dataframe(
        epsilons,
        alphas_betas,
        algorithm_names,
        results_dict,
        mu0,
        mu1,
        filename=csv_file,
    )

    end_time = datetime.now()
    print(f"\nTotal execution time: {end_time - start_time}")

    return results_dict
