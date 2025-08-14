"""
Configuration module for SPRT experiment parameters and settings.

Simplified version of the original config.py file, focused on just what's needed
for running SPRT comparison experiments.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Set, ClassVar
import json
import hashlib


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.

    This class encapsulates all parameters needed to define an experiment.

    Attributes:
        method: The SPRT variant to use (e.g., "SPRT", "DP-SPRT")
        mu0: Null hypothesis parameter (probability of success under H0)
        mu1: Alternative hypothesis parameter (probability of success under H1)
        alpha: Type I error rate (probability of rejecting H0 when it's true)
        beta: Type II error rate (probability of accepting H0 when H1 is true)
        epsilon: Privacy parameter (only for DP variants)
        c1: First threshold parameter (only for Tuned DP-SPRT)
        c2: Second threshold parameter (only for Tuned DP-SPRT)
        n_sims: Number of simulations to run
        max_samples: Maximum number of samples per sequence
        seed: Random seed for reproducibility
        delta: Delta parameter for differential privacy
        A: Truncation parameter for Priv-SPRT
    """

    # Required parameters
    method: str  # The SPRT variant to use
    mu0: float  # Null hypothesis parameter
    mu1: float  # Alternative hypothesis parameter
    alpha: float  # Type I error rate
    beta: float  # Type II error rate

    # Optional parameters
    epsilon: Optional[float] = None  # Privacy parameter (for DP variants)
    c1: Optional[float] = None  # First threshold parameter (for Tuned DP-SPRT)
    c2: Optional[float] = None  # Second threshold parameter (for Tuned DP-SPRT)
    n_sims: int = 1000  # Number of simulations
    max_samples: int = 1000  # Maximum samples per sequence
    seed: int = 42  # Random seed for reproducibility
    delta: Optional[float] = None  # Delta parameter for DP
    A: Optional[float] = None  # Truncation parameter for Priv-SPRT

    # Class variables for validation
    REQUIRED_PARAMS: ClassVar[Set[str]] = {"method", "mu0", "mu1", "alpha", "beta"}

    METHOD_PARAMS: ClassVar[Dict[str, Set[str]]] = {
        "DP-SPRT": {"epsilon"},
        "Tuned DP-SPRT": {"epsilon", "c1", "c2"},
        "Priv-SPRT": {"epsilon", "delta", "A"},
    }

    def validate(self) -> bool:
        """Validate the configuration.

        Checks that all required parameters are present and valid for the
        specified method.

        Returns:
            True if the configuration is valid

        Raises:
            ValueError: If any validation check fails
        """
        # Validate basic parameters
        if not 0 < self.mu0 < 1:
            raise ValueError(f"mu0 must be between 0 and 1, got {self.mu0}")
        if not 0 < self.mu1 < 1:
            raise ValueError(f"mu1 must be between 0 and 1, got {self.mu1}")
        if self.mu0 == self.mu1:
            raise ValueError(f"mu0 and mu1 must be different, got both {self.mu0}")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if not 0 < self.beta < 1:
            raise ValueError(f"beta must be between 0 and 1, got {self.beta}")

        # Validate method-specific parameters
        if "DP" in self.method:
            if self.epsilon is None:
                raise ValueError(f"epsilon is required for {self.method}")
            if self.epsilon <= 0:
                raise ValueError(f"epsilon must be positive, got {self.epsilon}")

        if self.method == "Tuned DP-SPRT":
            if self.c1 is None:
                raise ValueError("c1 is required for Tuned DP-SPRT")
            if self.c2 is None:
                raise ValueError("c2 is required for Tuned DP-SPRT")

        if self.method == "Priv-SPRT":
            if self.delta is None:
                raise ValueError("delta is required for Priv-SPRT")
            if self.A is None:
                raise ValueError("A is required for Priv-SPRT")

        # Validate other parameters
        if self.n_sims <= 0:
            raise ValueError(f"n_sims must be positive, got {self.n_sims}")
        if self.max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {self.max_samples}")

        return True

    def get_id(self) -> str:
        """Generate a unique identifier for this experiment configuration.

        Returns:
            A string identifier for the configuration
        """
        # Create a dictionary of all non-None parameters
        params = {k: v for k, v in self.__dict__.items() if v is not None}

        # Convert to a stable string representation and hash it
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(param_str.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create a config from a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters

        Returns:
            An ExperimentConfig instance
        """
        return cls(**config_dict)


def create_experiment_suite(
    base_config: Dict[str, Any], variants: Dict[str, Dict[str, Any]]
) -> Dict[str, ExperimentConfig]:
    """Create a suite of experiments from a base configuration and variants.

    This function creates a collection of related experiment configurations by
    combining a base configuration with variant-specific parameters.

    Args:
        base_config: Base configuration parameters
        variants: Dictionary mapping variant names to their specific parameters

    Returns:
        Dictionary mapping variant names to ExperimentConfig objects
    """
    experiments = {}
    for name, variant_params in variants.items():
        # Merge base config with variant-specific parameters
        config_dict = base_config.copy()
        config_dict.update(variant_params)

        # Create the configuration
        experiments[name] = ExperimentConfig.from_dict(config_dict)
    return experiments
