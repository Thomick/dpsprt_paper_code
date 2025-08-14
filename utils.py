"""
Common utilities and helper functions for SPRT experiments.

Simplified version of the original utils.py file, focused on just what's needed
for running SPRT comparison experiments.
"""

import logging
import os
import json
from typing import Dict, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np


def ensure_dir_exists(path: Union[str, Path]) -> None:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path
    """
    if not path:
        return

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def convert_to_serializable(obj: Any) -> Any:
    """Convert non-serializable objects to JSON-serializable format.

    Args:
        obj: The object to convert

    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_json(
    data: Dict[str, Any],
    file_path: Union[str, Path],
    indent: int = 2,
    ensure_dir: bool = True,
) -> None:
    """Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to the JSON file
        indent: Indentation level for pretty-printing
        ensure_dir: Whether to ensure the directory exists

    Raises:
        IOError: If the file cannot be written
    """
    if ensure_dir:
        ensure_dir_exists(Path(file_path).parent)

    with open(file_path, "w") as f:
        json.dump(convert_to_serializable(data), f, indent=indent)


def save_results_to_dataframe(
    epsilons, alphas_betas, algorithm_names, results_dict, mu0, mu1, filename="results.csv"
):
    """Convert results to a pandas DataFrame and save to CSV.

    Args:
        epsilons: List of epsilon values
        alphas_betas: List of alpha=beta values
        algorithm_names: List of algorithm names
        results_dict: Dictionary containing results
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        filename: Output CSV filename

    Returns:
        DataFrame containing the results
    """
    # Create list to store rows
    rows = []

    # For each combination of parameters
    for alpha_beta in alphas_betas:
        for i, epsilon in enumerate(epsilons):
            row = {
                "epsilon": epsilon,
                "alpha_beta": alpha_beta,
                "mu0": mu0,
                "mu1": mu1,
            }

            # Add metrics for each algorithm
            for algorithm in algorithm_names:
                if algorithm in results_dict and alpha_beta in results_dict[algorithm]:
                    if i < len(results_dict[algorithm][alpha_beta]):
                        result = results_dict[algorithm][alpha_beta][i]
                        for metric in ["avg_sample_size", "type_i_rate", "type_ii_rate"]:
                            if metric in result:
                                row[f"{algorithm}_{metric}"] = result[metric]
                            else:
                                row[f"{algorithm}_{metric}"] = np.nan
                    else:
                        for metric in ["avg_sample_size", "type_i_rate", "type_ii_rate"]:
                            row[f"{algorithm}_{metric}"] = np.nan
                else:
                    for metric in ["avg_sample_size", "type_i_rate", "type_ii_rate"]:
                        row[f"{algorithm}_{metric}"] = np.nan

            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    ensure_dir_exists(Path(filename).parent)
    df.to_csv(filename, index=False)
    
    return df
