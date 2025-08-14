"""
Sequential Probability Ratio Test (SPRT) algorithm implementations.

This module provides classes and factory functions for various SPRT algorithms,
including classical SPRT and differentially private variants.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Any, Optional, Union, List
import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np

from core import (
    sprt_test_core,
    dp_sprt_test_core,
    optimized_dp_sprt_test_core,
    dp_sprt_test_subsampled_core,
    priv_sprt_test_core,
    find_optimal_thresholds,
    dp_sprt_gaussian_test_core,
)


@dataclass
class SPRTParameters:
    """Common parameters for SPRT algorithms.

    Attributes:
        mu0: Null hypothesis parameter (probability of success under H0)
        mu1: Alternative hypothesis parameter (probability of success under H1)
        alpha: Type I error rate (probability of rejecting H0 when it's true)
        beta: Type II error rate (probability of accepting H0 when H1 is true)
    """

    mu0: float
    mu1: float
    alpha: float
    beta: float

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()

    def validate(self):
        """Validate parameter values."""
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
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        return {
            "mu0": self.mu0,
            "mu1": self.mu1,
            "alpha": self.alpha,
            "beta": self.beta,
        }


class SPRT:
    """Base class for Sequential Probability Ratio Test algorithms."""

    def __init__(self, params: SPRTParameters):
        """Initialize with algorithm parameters."""
        self.params = params

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run the test on a sequence of observations."""
        raise NotImplementedError("Subclasses must implement test method")

    def batch_test(
        self,
        data: jnp.ndarray,
        keys: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a batch of tests on multiple sequences."""
        raise NotImplementedError("Subclasses must implement batch_test method")


class ClassicalSPRT(SPRT):
    """Classical Sequential Probability Ratio Test."""

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run the classical SPRT on a sequence of observations."""
        decision, stopping_time, additional_info = sprt_test_core(
            x, self.params.mu0, self.params.mu1, self.params.alpha, self.params.beta
        )

        if return_details:
            return {
                "decision": decision,
                "stopping_time": stopping_time,
                **additional_info,
            }
        else:
            return decision, stopping_time

    def batch_test(
        self,
        data: jnp.ndarray,
        keys: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a batch of classical SPRT tests on multiple sequences."""
        batch_fn = lambda x: self.test(x, None, return_details)
        results = vmap(batch_fn)(data)

        if return_details:
            # Convert JAX arrays to numpy arrays after vmap
            return {
                k: np.array(v) if isinstance(v, jnp.ndarray) else v
                for k, v in results.items()
            }
        else:
            # Handle tuple results from vmap
            decisions, stopping_times = results
            return {
                "decisions": np.array(decisions),
                "stopping_times": np.array(stopping_times),
            }


@dataclass
class DPSPRTParameters(SPRTParameters):
    """Parameters for differentially private SPRT algorithms.

    Attributes:
        epsilon: Privacy parameter
    """

    epsilon: float

    def validate(self):
        """Validate parameter values."""
        super().validate()
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        params = super().to_dict()
        params["epsilon"] = self.epsilon
        return params


class DPSPRT(SPRT):
    """Differentially Private Sequential Probability Ratio Test."""

    def __init__(self, params: DPSPRTParameters):
        """Initialize with DP-SPRT parameters."""
        super().__init__(params)
        self.epsilon = params.epsilon

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run DP-SPRT on a sequence of observations."""
        if key is None:
            raise ValueError("Random key is required for DP-SPRT")

        decision, stopping_time, additional_info = dp_sprt_test_core(
            x,
            self.params.mu0,
            self.params.mu1,
            self.params.alpha,
            self.params.beta,
            self.epsilon,
            key,
        )

        if return_details:
            return {
                "decision": decision,
                "stopping_time": stopping_time,
                **additional_info,
            }
        else:
            return decision, stopping_time

    def batch_test(
        self,
        data: jnp.ndarray,
        keys: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a batch of DP-SPRT tests on multiple sequences."""
        if keys is None:
            raise ValueError("Random keys are required for DP-SPRT batch tests")

        batch_fn = lambda x, k: self.test(x, k, return_details)
        results = vmap(batch_fn)(data, keys)

        if return_details:
            # Convert JAX arrays to numpy arrays after vmap
            if isinstance(results, dict):
                return {
                    k: np.array(v) if isinstance(v, jnp.ndarray) else v
                    for k, v in results.items()
                }
            else:
                # Handle unexpected output type (should be a dict)
                raise TypeError(f"Expected dict from vmap, got {type(results)}")
        else:
            # Handle tuple results from vmap
            decisions, stopping_times = results
            return {
                "decisions": np.array(decisions),
                "stopping_times": np.array(stopping_times),
            }


class DPSPRTSubsampled(DPSPRT):
    """Differentially Private SPRT with subsampling."""

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run DP-SPRT with subsampling on a sequence of observations."""
        if key is None:
            raise ValueError("Random key is required for DP-SPRT with subsampling")

        n = len(x)
        q = jnp.minimum(1.0, jnp.sqrt(self.epsilon / 10))

        # Split keys
        key_random, key_rest = random.split(key)

        # Generate inclusion decisions for each element with fixed probability q
        element_randoms = random.uniform(key_random, shape=(n,))
        include_element = (
            element_randoms < q
        )  # Boolean array: which elements to subsample

        # Fast vectorized computation using cumulative operations
        # Create masked version of x where non-included elements are 0
        x_masked = jnp.where(include_element, x, 0.0)
        include_count = include_element.astype(jnp.float32)

        # Compute cumulative sums and counts of included elements
        cum_sum_included = jnp.cumsum(x_masked)
        cum_count_included = jnp.cumsum(include_count)

        # For each step i, we want the mean of included elements among the first i+1 elements
        steps = jnp.arange(1, n + 1)  # Step indices: 1, 2, 3, ..., n

        # Compute means at each step (with fallback for when no elements are included)
        means = jnp.where(
            cum_count_included > 0,
            cum_sum_included / cum_count_included,
            jnp.cumsum(x) / steps,  # Fallback: use all elements if none subsampled
        )

        means = means.reshape(-1, 1)

        # Run the core DP-SPRT test
        decision, stopping_time, additional_info = dp_sprt_test_subsampled_core(
            means,
            self.params.mu0,
            self.params.mu1,
            self.params.alpha,
            self.params.beta,
            self.epsilon,
            key_rest,
            q=q,
        )

        if return_details:
            return {
                "decision": decision,
                "stopping_time": stopping_time,
                **additional_info,
            }
        else:
            return decision, stopping_time


@dataclass
class TunedDPSPRTParameters(DPSPRTParameters):
    """Parameters for Tuned DP-SPRT algorithm.

    Attributes:
        c1: First threshold optimization parameter
        c2: Second threshold optimization parameter
    """

    c1: float = 1.0
    c2: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        params = super().to_dict()
        params.update({"c1": self.c1, "c2": self.c2})
        return params


class TunedDPSPRT(SPRT):
    """Tuned Differentially Private Sequential Probability Ratio Test."""

    def __init__(self, params: TunedDPSPRTParameters):
        """Initialize with Tuned DP-SPRT parameters."""
        super().__init__(params)
        self.epsilon = params.epsilon
        self.c1 = params.c1
        self.c2 = params.c2

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run tuned DP-SPRT on a sequence of observations."""
        if key is None:
            raise ValueError("Random key is required for Tuned DP-SPRT")

        decision, stopping_time, additional_info = optimized_dp_sprt_test_core(
            x,
            self.params.mu0,
            self.params.mu1,
            self.params.alpha,
            self.params.beta,
            self.epsilon,
            key,
            self.c1,
            self.c2,
        )

        if return_details:
            return {
                "decision": decision,
                "stopping_time": stopping_time,
                **additional_info,
            }
        else:
            return decision, stopping_time

    def batch_test(
        self,
        data: jnp.ndarray,
        keys: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a batch of tuned DP-SPRT tests on multiple sequences."""
        if keys is None:
            raise ValueError("Random keys are required for Tuned DP-SPRT batch tests")

        batch_fn = lambda x, k: self.test(x, k, return_details)
        results = vmap(batch_fn)(data, keys)

        if return_details:
            # Convert JAX arrays to numpy arrays after vmap
            if isinstance(results, dict):
                return {
                    k: np.array(v) if isinstance(v, jnp.ndarray) else v
                    for k, v in results.items()
                }
            else:
                # Handle unexpected output type (should be a dict)
                raise TypeError(f"Expected dict from vmap, got {type(results)}")
        else:
            # Handle tuple results from vmap
            decisions, stopping_times = results
            return {
                "decisions": np.array(decisions),
                "stopping_times": np.array(stopping_times),
            }


@dataclass
class PrivSPRTParameters(DPSPRTParameters):
    """Parameters for PrivSPRT algorithm.

    Attributes:
        delta: Privacy failure probability
        A: Truncation parameter for log-likelihood ratios
        threshold_a: Optional pre-computed lower threshold value
        threshold_b: Optional pre-computed upper threshold value
    """

    delta: float = 1e-5
    A: float = 1.0
    threshold_a: Optional[float] = None
    threshold_b: Optional[float] = None

    def validate(self):
        """Validate parameter values."""
        super().validate()
        if self.delta <= 0 or self.delta >= 1:
            raise ValueError(f"delta must be between 0 and 1, got {self.delta}")
        if self.A <= 0:
            raise ValueError(f"A must be positive, got {self.A}")

        # If thresholds are provided, validate that a < b
        if self.threshold_a is not None and self.threshold_b is not None:
            if self.threshold_a >= self.threshold_b:
                raise ValueError(
                    f"threshold_a must be less than threshold_b, got {self.threshold_a} >= {self.threshold_b}"
                )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        params = super().to_dict()
        params.update({"delta": self.delta, "A": self.A})
        if self.threshold_a is not None:
            params["threshold_a"] = self.threshold_a
        if self.threshold_b is not None:
            params["threshold_b"] = self.threshold_b
        return params


class PrivSPRT(SPRT):
    """PrivSPRT (Truncated Log-Likelihood Ratio with Gaussian noise)."""

    def __init__(self, params: PrivSPRTParameters):
        """Initialize with PrivSPRT parameters."""
        super().__init__(params)
        self.epsilon = params.epsilon
        self.delta = params.delta
        self.A = params.A

        # Either use provided thresholds or compute optimal thresholds
        if params.threshold_a is not None and params.threshold_b is not None:
            self.threshold_a = params.threshold_a
            self.threshold_b = params.threshold_b
        else:
            self.threshold_a, self.threshold_b = find_optimal_thresholds(
                params.mu0,
                params.mu1,
                params.alpha,
                params.beta,
                params.epsilon,
                params.delta,
                params.A,
                num_trials=100,
                max_samples=10000,
            )
            # Update params with computed thresholds
            params.threshold_a = self.threshold_a
            params.threshold_b = self.threshold_b

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run PrivSPRT on a sequence of observations."""
        if key is None:
            raise ValueError("Random key is required for PrivSPRT")

        decision, stopping_time, additional_info = priv_sprt_test_core(
            x,
            self.params.mu0,
            self.params.mu1,
            self.params.alpha,
            self.params.beta,
            self.epsilon,
            self.delta,
            self.A,
            self.threshold_a,
            self.threshold_b,
            key,
        )

        if return_details:
            return {
                "decision": decision,
                "stopping_time": stopping_time,
                **additional_info,
            }
        else:
            return decision, stopping_time

    def batch_test(
        self,
        data: jnp.ndarray,
        keys: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """Run a batch of PrivSPRT tests on multiple sequences."""
        if keys is None:
            raise ValueError("Random keys are required for PrivSPRT batch tests")

        batch_fn = lambda x, k: self.test(x, k, return_details)
        results = vmap(batch_fn)(data, keys)

        if return_details:
            # Convert JAX arrays to numpy arrays after vmap
            if isinstance(results, dict):
                return {
                    k: np.array(v) if isinstance(v, jnp.ndarray) else v
                    for k, v in results.items()
                }
            else:
                # Handle unexpected output type (should be a dict)
                raise TypeError(f"Expected dict from vmap, got {type(results)}")
        else:
            # Handle tuple results from vmap
            decisions, stopping_times = results
            return {
                "decisions": np.array(decisions),
                "stopping_times": np.array(stopping_times),
            }


@dataclass
class DPSPRTGaussianParams(DPSPRTParameters):
    """Parameters for DP-SPRT with Gaussian noise.

    Attributes:
        delta: Privacy parameter for (epsilon, delta)-DP
    """

    delta: float

    def validate(self):
        """Validate parameter values."""
        super().validate()
        if not 0 < self.delta < 1:
            raise ValueError(f"delta must be between 0 and 1, got {self.delta}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to a dictionary."""
        params = super().to_dict()
        params["delta"] = self.delta
        return params


class DPSPRTGaussian(DPSPRT):  # Inherits from DPSPRT for batch_test reuse
    """Differentially Private SPRT with Gaussian Noise."""

    def __init__(self, params: DPSPRTGaussianParams):
        """Initialize with DP-SPRT Gaussian parameters."""
        super().__init__(params)  # Calls DPSPRT init, which sets self.epsilon
        self.delta = params.delta  # Explicitly set delta

    def test(
        self,
        x: jnp.ndarray,
        key: Optional[jnp.ndarray] = None,
        return_details: bool = False,
    ) -> Union[Tuple[jnp.ndarray, jnp.ndarray], Dict[str, Any]]:
        """Run DP-SPRT with Gaussian noise on a sequence of observations."""
        if key is None:
            raise ValueError("Random key is required for DP-SPRT Gaussian")

        # Ensure self.params is DPSPRTGaussianParams for type hinting if needed, though access is via self.epsilon and self.delta
        # The core function expects epsilon and delta directly.
        decision, stopping_time, additional_info = dp_sprt_gaussian_test_core(
            x,
            self.params.mu0,
            self.params.mu1,
            self.params.alpha,
            self.params.beta,
            self.epsilon,  # from DPSPRT parent
            self.delta,  # specific to this class
            key,
        )

        if return_details:
            return {
                "decision": decision,
                "stopping_time": stopping_time,
                **additional_info,
            }
        else:
            return decision, stopping_time

    # batch_test is inherited from DPSPRT, which correctly uses the overridden test method.


# Factory functions for easier instantiation


def create_classical_sprt(
    mu0: float, mu1: float, alpha: float, beta: float
) -> ClassicalSPRT:
    """Create a ClassicalSPRT instance with the given parameters."""
    params = SPRTParameters(mu0=mu0, mu1=mu1, alpha=alpha, beta=beta)
    return ClassicalSPRT(params)


def create_dp_sprt(
    mu0: float, mu1: float, alpha: float, beta: float, epsilon: float
) -> DPSPRT:
    """Create a DPSPRT instance with the given parameters."""
    params = DPSPRTParameters(mu0=mu0, mu1=mu1, alpha=alpha, beta=beta, epsilon=epsilon)
    return DPSPRT(params)


def create_dp_sprt_subsampled(
    mu0: float, mu1: float, alpha: float, beta: float, epsilon: float
) -> DPSPRTSubsampled:
    """Create a DPSPRTSubsampled instance with the given parameters."""
    params = DPSPRTParameters(mu0=mu0, mu1=mu1, alpha=alpha, beta=beta, epsilon=epsilon)
    return DPSPRTSubsampled(params)


def create_tuned_dp_sprt(
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    c1: float = 1.0,
    c2: float = 1.0,
) -> TunedDPSPRT:
    """Create a TunedDPSPRT instance with the given parameters."""
    params = TunedDPSPRTParameters(
        mu0=mu0, mu1=mu1, alpha=alpha, beta=beta, epsilon=epsilon, c1=c1, c2=c2
    )
    return TunedDPSPRT(params)


def create_priv_sprt(
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    delta: float,
    A: float = 1.0,
    threshold_a: Optional[float] = None,
    threshold_b: Optional[float] = None,
) -> PrivSPRT:
    """Create a PrivSPRT instance with the given parameters."""
    params = PrivSPRTParameters(
        mu0=mu0,
        mu1=mu1,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        delta=delta,
        A=A,
        threshold_a=threshold_a,
        threshold_b=threshold_b,
    )
    return PrivSPRT(params)


def create_dp_sprt_gaussian(
    mu0: float, mu1: float, alpha: float, beta: float, epsilon: float, delta: float
) -> DPSPRTGaussian:
    """Create a DPSPRTGaussian instance with the given parameters."""
    params = DPSPRTGaussianParams(
        mu0=mu0, mu1=mu1, alpha=alpha, beta=beta, epsilon=epsilon, delta=delta
    )
    return DPSPRTGaussian(params)
