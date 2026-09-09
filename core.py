"""
Core implementations of Sequential Probability Ratio Test (SPRT) variants.

This module contains pure function implementations of various SPRT algorithms,
optimized for JAX JIT compilation. These functions form the computational backbone
of the library.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
from functools import partial
from typing import Dict, Tuple, Any, Optional, Union, Callable
import numpy as np

# Configure JAX for 64-bit precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# zeta(1.1340), the value paired with s = 1.1340 in the correction function.
# This was 5.591, which is zeta(1.2000), so the union bound over n in the
# correctness proof spent 1.44 delta instead of delta. Correcting it lengthens
# mean stopping times by about 4 percent.
ZETA_S_VALUE = 8.049572


@jit
def sprt_test_core(
    x: jnp.ndarray, mu0: float, mu1: float, alpha: float, beta: float
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Core implementation of classical SPRT algorithm for Bernoulli samples.

    Args:
        x: Array of Bernoulli observations
        mu0: Null hypothesis parameter (probability of success under H0)
        mu1: Alternative hypothesis parameter (probability of success under H1)
        alpha: Type I error rate (probability of rejecting H0 when it's true)
        beta: Type II error rate (probability of accepting H0 when H1 is true)

    Returns:
        Tuple containing:
            decision: -1 (accept H0), 1 (accept H1), or 0 (no decision)
            stop_time: Number of samples needed for decision
            additional_info: Dictionary with additional statistics
    """
    cum_sum = jnp.cumsum(x)
    n = jnp.arange(1, len(x) + 1)

    # Log likelihood ratio for Bernoulli
    llr = cum_sum * jnp.log(mu1 / mu0) + (n - cum_sum) * jnp.log((1 - mu1) / (1 - mu0))

    # Decision boundaries
    upper = jnp.log(1 / alpha)
    lower = jnp.log(beta)

    crossed_upper = llr >= upper
    crossed_lower = llr <= lower
    crossed_either = crossed_upper | crossed_lower

    first_cross = jnp.argmax(crossed_either)
    no_decision = ~jnp.any(crossed_either)

    stop_time = jnp.where(no_decision, len(x), first_cross + 1)
    decision = jnp.where(no_decision, 0, jnp.where(crossed_upper[first_cross], 1, -1))

    # Additional statistics for detailed analysis
    additional_info = {
        "llr": llr,
        "cum_sum": cum_sum,
        "upper_threshold": upper,
        "lower_threshold": lower,
        "crossed_upper": crossed_upper,
        "crossed_lower": crossed_lower,
        "first_cross": first_cross,
        "no_decision": no_decision,
    }

    return decision, stop_time, additional_info


@jit
def dp_sprt_test_core(
    x: jnp.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    key: jnp.ndarray,
    subsampling_rate: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Core implementation of DP-SPRT with privacy-preserving stopping criterion.

    Args:
        x: Array of Bernoulli observations
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        alpha: Type I error rate
        beta: Type II error rate
        epsilon: Privacy parameter
        key: JAX random key for generating noise
        q: Subsample rate (default: 1.0)

    Returns:
        Tuple containing:
            decision: -1 (accept H0), 1 (accept H1), or 0 (no decision)
            stop_time: Number of samples needed for decision
            additional_info: Dictionary with additional statistics
    """
    s = 1.1340  # Parameter for zeta function
    zeta_s = ZETA_S_VALUE  # Pre-computed value of zeta(s)

    cum_sum = jnp.cumsum(x)
    n = jnp.arange(1, len(x) + 1, dtype=jnp.float64)
    mean = cum_sum / n

    # Add Laplace noise for privacy
    key1, key2, key3 = random.split(key, 3)
    lap1 = random.laplace(key1, shape=mean.shape) * (4 / (n * epsilon))
    lap2 = random.laplace(key2) * (2 / (n * epsilon))

    # Compute exponential family parameters
    def theta(mu):
        p = jnp.clip(mu, 1e-10, 1 - 1e-10)
        return jnp.log((p / (1 - p)))

    def kl_div(p, q):
        p = jnp.clip(p, 1e-10, 1 - 1e-10)
        q = jnp.clip(q, 1e-10, 1 - 1e-10)
        return p * jnp.log(p / q) + (1 - p) * jnp.log((1 - p) / (1 - q))

    # Privacy-preserving stopping conditions
    theta0 = theta(mu0)
    theta1 = theta(mu1)

    tau0_cond = mean - mu0 + lap1 <= (
        kl_div(mu0, mu1)
        - jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon)) * beta)) / (n)
    ) / (theta1 - theta0) - lap2 - 6 * jnp.log(
        jnp.maximum(epsilon, 2) * n**s * zeta_s / beta
    ) / (
        n * epsilon
    )

    tau1_cond = mean - mu1 + lap1 >= (
        -kl_div(mu1, mu0)
        + jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon)) * alpha)) / (n)
    ) / (theta1 - theta0) + lap2 + 6 * jnp.log(
        jnp.maximum(epsilon, 2) * n**s * zeta_s / alpha
    ) / (
        n * epsilon
    )

    # Compute stopping times
    tau0 = jnp.where(jnp.any(tau0_cond), jnp.argmax(tau0_cond) + 1, len(x))
    tau1 = jnp.where(jnp.any(tau1_cond), jnp.argmax(tau1_cond) + 1, len(x))

    stop_time = jnp.minimum(tau0, tau1)
    decision = jnp.where(stop_time == tau0, -1, 1)
    decision = jnp.where(stop_time == len(x), 0, decision)

    # Additional statistics for detailed analysis
    additional_info = {
        "cum_sum": cum_sum,
        "mean": mean,
        "noisy_mean": mean + lap1,  # Privatized mean
        "lap1": lap1,
        "lap2": lap2,
        "lap3": lap2,
        "theta0": theta0,
        "theta1": theta1,
        "tau0_cond": tau0_cond,
        "tau1_cond": tau1_cond,
        "tau0": tau0,
        "tau1": tau1,
        "kl_div_01": kl_div(mu0, mu1),
        "kl_div_10": kl_div(mu1, mu0),
    }

    return decision, stop_time, additional_info


@jit
def optimized_dp_sprt_test_core(
    x: jnp.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    key: jnp.ndarray,
    c1: float = 1.0,
    c2: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Core implementation of DP-SPRT with optimized privacy-preserving stopping criterion.

    Args:
        x: Array of Bernoulli observations
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        alpha: Type I error rate
        beta: Type II error rate
        epsilon: Privacy parameter
        key: JAX random key for generating noise
        c1: First threshold optimization parameter (default: 1.0)
        c2: Second threshold optimization parameter (default: 1.0)

    Returns:
        Tuple containing:
            decision: -1 (accept H0), 1 (accept H1), or 0 (no decision)
            stop_time: Number of samples needed for decision
            additional_info: Dictionary with additional statistics
    """
    cum_sum = jnp.cumsum(x)
    n = jnp.arange(1, len(x) + 1, dtype=jnp.float64)
    mean = cum_sum / n

    # Add Laplace noise for privacy
    key1, key2, key3 = random.split(key, 3)
    lap1 = random.laplace(key1, shape=mean.shape) * (4 / (n * epsilon))
    lap2 = random.laplace(key2) * (2 / (n * epsilon))

    # Compute exponential family parameters
    def theta(mu):
        p = jnp.clip(mu, 1e-10, 1 - 1e-10)
        return jnp.log((p / (1 - p)))

    def kl_div(p, q):
        p = jnp.clip(p, 1e-10, 1 - 1e-10)
        q = jnp.clip(q, 1e-10, 1 - 1e-10)
        return p * jnp.log(p / q) + (1 - p) * jnp.log((1 - p) / (1 - q))

    # Privacy-preserving stopping conditions with optimization parameters
    theta0 = theta(mu0)
    theta1 = theta(mu1)
    s = 1.1340
    zeta_s = ZETA_S_VALUE

    tau0_cond = mean - mu0 + lap1 <= (
        kl_div(mu0, mu1)
        - c1 * jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon)) * beta)) / (n)
    ) / (theta1 - theta0) - lap2 - c2 * 6 * jnp.log(
        jnp.maximum(epsilon, 2) * n**s * zeta_s / beta
    ) / (
        n * epsilon
    )

    tau1_cond = mean - mu1 + lap1 >= (
        -kl_div(mu1, mu0)
        + c1 * jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon)) * alpha)) / (n)
    ) / (theta1 - theta0) + lap2 + c2 * 6 * jnp.log(
        jnp.maximum(epsilon, 2) * n**s * zeta_s / alpha
    ) / (
        n * epsilon
    )

    # Compute stopping times
    tau0 = jnp.where(jnp.any(tau0_cond), jnp.argmax(tau0_cond) + 1, len(x))
    tau1 = jnp.where(jnp.any(tau1_cond), jnp.argmax(tau1_cond) + 1, len(x))

    stop_time = jnp.minimum(tau0, tau1)
    decision = jnp.where(stop_time == tau0, -1, 1)
    decision = jnp.where(stop_time == len(x), 0, decision)

    # Additional statistics for detailed analysis
    additional_info = {
        "cum_sum": cum_sum,
        "mean": mean,
        "noisy_mean": mean + lap1,
        "lap1": lap1,
        "lap2": lap2,
        "lap3": lap2,
        "theta0": theta0,
        "theta1": theta1,
        "c1": c1,
        "c2": c2,
        "tau0_cond": tau0_cond,
        "tau1_cond": tau1_cond,
        "tau0": tau0,
        "tau1": tau1,
        "kl_div_01": kl_div(mu0, mu1),
        "kl_div_10": kl_div(mu1, mu0),
    }

    return decision, stop_time, additional_info


@jit
def dp_sprt_test_subsampled_core(
    means: jnp.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    key: jnp.ndarray,
    q: float = 1.0,  # Subsampling rate
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Core implementation of DP-SPRT with subsampling.

    This version uses pre-computed subsample means with privacy protection.

    Args:
        means: Array of subsampled means (shape: [n, 1] or [n,])
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        alpha: Type I error rate
        beta: Type II error rate
        epsilon: Privacy parameter
        key: JAX random key for generating noise
        q: Subsampling rate

    Returns:
        Tuple containing:
            decision: -1 (accept H0), 1 (accept H1), or 0 (no decision)
            stop_time: Number of samples needed for decision
            additional_info: Dictionary with additional statistics
    """
    s = 1.1340  # Parameter for zeta function
    zeta_s = ZETA_S_VALUE  # Pre-computed value of zeta(s)

    # Flatten means if it's 2D
    means_flat = means.flatten()
    n = len(means_flat)
    n_array = jnp.arange(1, n + 1, dtype=jnp.float64)

    # Add Laplace noise for privacy with subsampling adjustment
    key1, key2, key3 = random.split(key, 3)
    lap1 = random.laplace(key1, shape=means_flat.shape) * (q * 4 / (n_array * epsilon))
    lap2 = random.laplace(key2) * (2 * q / (n_array * epsilon))

    # Compute exponential family parameters
    def theta(mu):
        p = jnp.clip(mu, 1e-10, 1 - 1e-10)
        return jnp.log((p / (1 - p)))

    def kl_div(p, q_val):  # Renamed q to q_val to avoid conflict with subsampling rate
        p = jnp.clip(p, 1e-10, 1 - 1e-10)
        q_val = jnp.clip(q_val, 1e-10, 1 - 1e-10)
        return p * jnp.log(p / q_val) + (1 - p) * jnp.log((1 - p) / (1 - q_val))

    # Privacy-preserving stopping conditions with subsampling
    theta0 = theta(mu0)
    theta1 = theta(mu1)

    tau0_cond = means_flat - mu0 + lap1 <= (
        kl_div(mu0, mu1)
        - jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon)) * beta)) / (q * n_array)
    ) / (theta1 - theta0) - lap2 - 6 * q * jnp.log(
        jnp.maximum(epsilon, 2) * n_array**s * zeta_s / beta
    ) / (
        n_array * epsilon
    )

    tau1_cond = means_flat - mu1 + lap1 >= (
        -kl_div(mu1, mu0)
        + jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon)) * alpha)) / (q * n_array)
    ) / (theta1 - theta0) + lap2 + 6 * q * jnp.log(
        jnp.maximum(epsilon, 2) * n_array**s * zeta_s / alpha
    ) / (
        n_array * epsilon
    )

    # Compute stopping times
    tau0 = jnp.where(jnp.any(tau0_cond), jnp.argmax(tau0_cond) + 1, n)
    tau1 = jnp.where(jnp.any(tau1_cond), jnp.argmax(tau1_cond) + 1, n)

    stop_time = jnp.minimum(tau0, tau1)
    decision = jnp.where(stop_time == tau0, -1, 1)
    decision = jnp.where(stop_time == n, 0, decision)

    # Additional statistics for detailed analysis
    additional_info = {
        "means": means_flat,
        "noisy_means": means_flat + lap1,  # Privatized means
        "lap1": lap1,
        "lap2": lap2,
        "lap3": lap2,
        "theta0": theta0,
        "theta1": theta1,
        "tau0_cond": tau0_cond,
        "tau1_cond": tau1_cond,
        "tau0": tau0,
        "tau1": tau1,
        "kl_div_01": kl_div(mu0, mu1),
        "kl_div_10": kl_div(mu1, mu0),
        "q": q,  # Subsampling rate
    }

    return decision, stop_time, additional_info


@jit
def priv_sprt_test_core(
    x: jnp.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    delta: float,
    A: float,
    a: float,  # Lower threshold
    b: float,  # Upper threshold
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Core implementation of PrivSPRT with truncated log-likelihood ratios and Gaussian noise.

    Args:
        x: Array of Bernoulli observations
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        alpha: Type I error rate
        beta: Type II error rate
        epsilon: Privacy parameter
        delta: Privacy failure probability
        A: Truncation parameter
        a: Lower threshold
        b: Upper threshold
        key: JAX random key

    Returns:
        Tuple containing:
            decision: -1 (accept H0), 1 (accept H1), or 0 (no decision)
            stop_time: Number of samples needed for decision
            additional_info: Dictionary with additional statistics
    """
    # Apply the epsilon/2 adjustment to match the original implementation
    epsilon_prime = epsilon / (2 * jnp.sqrt(2))

    # Calculate Gaussian mechanism parameters
    sigma1 = jnp.sqrt(32 * jnp.log(1.25 / delta) * A**2) / epsilon_prime
    sigma2 = jnp.sqrt(128 * jnp.log(1.25 / delta) * A**2) / epsilon_prime

    # Split keys for different random operations
    keys = random.split(key, 3)
    key_threshold_a = keys[0]
    key_threshold_b = keys[1]
    key_llr = keys[2]

    # Add noise to thresholds
    noisy_a = a + random.normal(key_threshold_a) * sigma1
    noisy_b = b + random.normal(key_threshold_b) * sigma1

    # Function to calculate truncated log-likelihood ratio for Bernoulli data
    def truncated_llr(xi):
        p0 = jnp.clip(mu0, 1e-10, 1 - 1e-10)
        p1 = jnp.clip(mu1, 1e-10, 1 - 1e-10)
        llr = xi * jnp.log(p1 / p0) + (1 - xi) * jnp.log((1 - p1) / (1 - p0))
        return jnp.clip(llr, -A, A)

    # Compute LLRs for all data points
    llrs = jax.vmap(truncated_llr)(x)

    # Compute cumulative LLRs
    cum_llrs = jnp.cumsum(llrs)
    n_samples = len(x)

    # Generate noise for cumulative LLRs
    key_a, key_b = random.split(key_llr)
    noise_a = random.normal(key_a, (n_samples,)) * sigma2
    noise_b = random.normal(key_b, (n_samples,)) * sigma2

    # Add noise to cumulative LLRs
    noisy_cum_llrs_a = cum_llrs + noise_a
    noisy_cum_llrs_b = cum_llrs + noise_b

    # Check stopping conditions for all time steps
    accept_h0_times = jnp.where(
        noisy_cum_llrs_a <= noisy_a, jnp.arange(1, n_samples + 1), n_samples + 1
    )
    reject_h0_times = jnp.where(
        noisy_cum_llrs_b >= noisy_b, jnp.arange(1, n_samples + 1), n_samples + 1
    )

    # Find the minimum stopping time for each decision
    accept_time = jnp.min(accept_h0_times)
    reject_time = jnp.min(reject_h0_times)

    # Determine the earliest stopping time and corresponding decision
    stop_time = jnp.minimum(accept_time, reject_time)
    decision = jnp.where(
        stop_time == accept_time, -1, jnp.where(stop_time == reject_time, 1, 0)
    )

    # Only update if we actually stopped
    valid_stop = stop_time <= n_samples
    decision = jnp.where(valid_stop, decision, 0)
    stop_time = jnp.where(valid_stop, stop_time, n_samples)

    # Additional statistics for detailed analysis
    additional_info = {
        "llrs": llrs,
        "cum_llrs": cum_llrs,
        "noisy_cum_llrs_a": noisy_cum_llrs_a,
        "noisy_cum_llrs_b": noisy_cum_llrs_b,
        "noisy_threshold_a": noisy_a,
        "noisy_threshold_b": noisy_b,
        "noise_a": noise_a,
        "noise_b": noise_b,
        "threshold_a": a,
        "threshold_b": b,
        "epsilon_prime": epsilon_prime,
        "accept_h0_times": accept_h0_times,
        "reject_h0_times": reject_h0_times,
        "valid_stop": valid_stop,
    }

    return decision, stop_time, additional_info


@jit
def test_thresholds_priv_sprt(
    data: jnp.ndarray,
    threshold: float,  # We'll derive a and b from this single parameter
    mu0: float,
    mu1: float,
    epsilon: float,
    delta: float,
    A: float,
    key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Test a single threshold value on a single sequence (derive a=-threshold, b=threshold).

    Args:
        data: Array of Bernoulli observations
        threshold: Threshold to test (we'll use a=-threshold, b=threshold)
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        epsilon: Privacy parameter
        delta: Privacy failure probability
        A: Truncation parameter
        key: JAX random key

    Returns:
        Tuple containing (decision, stop_time)
    """
    # Derive a and b from threshold
    a = -threshold
    b = threshold

    # Only return decision and stop_time, not the additional_info
    decision, stop_time, _ = priv_sprt_test_core(
        data, mu0, mu1, 0.0, 0.0, epsilon, delta, A, a, b, key
    )
    return decision, stop_time


def evaluate_single_threshold(
    threshold,
    data_h0,
    data_h1,
    keys_h0,
    keys_h1,
    mu0,
    mu1,
    alpha,
    beta,
    epsilon,
    delta,
    A,
    num_trials,
):
    """Evaluate a single threshold value."""
    # Use existing batch_test logic but for single threshold
    batch_test = jax.vmap(
        test_thresholds_priv_sprt, in_axes=(0, None, None, None, None, None, None, 0)
    )

    decisions_h0, _ = batch_test(
        data_h0, threshold, mu0, mu1, epsilon, delta, A, keys_h0
    )
    decisions_h1, _ = batch_test(
        data_h1, threshold, mu0, mu1, epsilon, delta, A, keys_h1
    )

    type_i_rate = jnp.sum(decisions_h0 == 1) / num_trials
    type_ii_rate = jnp.sum(decisions_h1 == -1) / num_trials

    return jnp.abs(type_i_rate - alpha) + jnp.abs(type_ii_rate - beta)


@partial(jit, static_argnames=["num_trials"])
def optimize_priv_sprt_thresholds(
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    delta: float,
    A: float,
    data_h0: jnp.ndarray,
    data_h1: jnp.ndarray,
    keys_h0: jnp.ndarray,
    keys_h1: jnp.ndarray,
    num_trials: int,
) -> Tuple[float, float]:
    """Coarse-to-fine search to efficiently find optimal threshold."""

    # Stage 1: Coarse search
    coarse_thresholds = jnp.linspace(1.0, 100.0 / (epsilon * alpha), 50)
    coarse_errors = jax.vmap(
        lambda t: evaluate_single_threshold(
            t,
            data_h0,
            data_h1,
            keys_h0,
            keys_h1,
            mu0,
            mu1,
            alpha,
            beta,
            epsilon,
            delta,
            A,
            num_trials,
        )
    )(coarse_thresholds)

    best_coarse_idx = jnp.argmin(coarse_errors)
    best_coarse_threshold = coarse_thresholds[best_coarse_idx]

    # Stage 2: Fine search around best coarse threshold
    search_radius = 5.0 / (epsilon * alpha)
    fine_min = jnp.maximum(1.0, best_coarse_threshold - search_radius)
    fine_max = best_coarse_threshold + search_radius
    fine_thresholds = jnp.linspace(fine_min, fine_max, 100)

    fine_errors = jax.vmap(
        lambda t: evaluate_single_threshold(
            t,
            data_h0,
            data_h1,
            keys_h0,
            keys_h1,
            mu0,
            mu1,
            alpha,
            beta,
            epsilon,
            delta,
            A,
            num_trials,
        )
    )(fine_thresholds)

    best_fine_idx = jnp.argmin(fine_errors)
    final_threshold = fine_thresholds[best_fine_idx]

    return -final_threshold, final_threshold


def find_optimal_thresholds(
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    delta: float,
    A: float,
    num_trials: int = 100,
    max_samples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Find optimal thresholds for PrivSPRT.

    Args:
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        alpha: Type I error rate
        beta: Type II error rate
        epsilon: Privacy parameter
        delta: Privacy failure probability
        A: Truncation parameter
        num_trials: Number of trials to use for optimization
        max_samples: Maximum number of samples per sequence
        seed: Random seed

    Returns:
        Tuple containing (a, b) - the lower and upper thresholds
    """
    # Initialize random key
    key = random.PRNGKey(seed)

    # Generate data for H0 and H1
    key, subkey = random.split(key)
    key_h0, key_h1 = random.split(subkey)
    data_h0 = random.bernoulli(key_h0, p=mu0, shape=(num_trials, max_samples))
    data_h1 = random.bernoulli(key_h1, p=mu1, shape=(num_trials, max_samples))

    # Generate test keys
    key, subkey = random.split(key)
    key_test_h0, key_test_h1 = random.split(subkey)
    keys_h0 = random.split(key_test_h0, num_trials)
    keys_h1 = random.split(key_test_h1, num_trials)

    try:
        # Find optimal thresholds
        a, b = optimize_priv_sprt_thresholds(
            mu0,
            mu1,
            alpha,
            beta,
            epsilon,
            delta,
            A,
            data_h0,
            data_h1,
            keys_h0,
            keys_h1,
            num_trials,
        )

        # Convert JAX arrays to Python floats
        return float(a), float(b)
    except Exception as e:
        # In case of error, fall back to default thresholds
        print(f"Error finding optimal thresholds: {e}")
        print("Using default thresholds instead")
        # For default thresholds, use a simple heuristic
        default_threshold = 10.0 / epsilon
        return -default_threshold, default_threshold


@jit
def dp_sprt_gaussian_test_core(
    x: jnp.ndarray,
    mu0: float,
    mu1: float,
    alpha: float,
    beta: float,
    epsilon: float,
    delta: float,  # New parameter
    key: jnp.ndarray,
    subsampling_rate: float = 1.0,  # Kept for consistency, though unused
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Core implementation of DP-SPRT with Gaussian noise.

    Args:
        x: Array of Bernoulli observations
        mu0: Null hypothesis parameter
        mu1: Alternative hypothesis parameter
        alpha: Type I error rate
        beta: Type II error rate
        epsilon: Privacy parameter (for (epsilon, delta)-DP)
        delta: Privacy parameter (for (epsilon, delta)-DP)
        key: JAX random key for generating noise
        subsampling_rate: Subsample rate (default: 1.0, currently unused in logic)

    Returns:
        Tuple containing:
            decision: -1 (accept H0), 1 (accept H1), or 0 (no decision)
            stop_time: Number of samples needed for decision
            additional_info: Dictionary with additional statistics
    """
    s = 1.1340  # Parameter for zeta function
    # Assuming ZETA_S_VALUE is defined in the global scope of core.py
    zeta_s = ZETA_S_VALUE

    cum_sum = jnp.cumsum(x)
    n = jnp.arange(1, len(x) + 1, dtype=jnp.float64)
    mean = cum_sum / n

    epsilon_prime = epsilon / 2
    # Add Gaussian noise for privacy ((epsilon, delta)-DP)
    key1, key2, _ = random.split(key, 3)  # key3 was unused in original, still unused

    gauss1_std_dev = jnp.sqrt(8 * jnp.log(1.25 / delta)) / (2 * epsilon_prime * n)
    gauss1 = random.normal(key1, shape=mean.shape) * gauss1_std_dev

    gauss2_std_dev = jnp.sqrt(2 * jnp.log(1.25 / delta)) / (2 * epsilon_prime * n)
    gauss2 = (
        random.normal(key2) * gauss2_std_dev
    )  # random.normal(key) is scalar, broadcasts with array n

    # Compute exponential family parameters
    def theta(
        mu_param,
    ):  # Renamed mu to mu_param to avoid conflict with outer scope if any
        p = jnp.clip(mu_param, 1e-10, 1 - 1e-10)
        return jnp.log((p / (1 - p)))

    def kl_div(p_param, q_param):  # Renamed p, q
        p_val = jnp.clip(p_param, 1e-10, 1 - 1e-10)
        q_val = jnp.clip(q_param, 1e-10, 1 - 1e-10)
        return p_val * jnp.log(p_val / q_val) + (1 - p_val) * jnp.log(
            (1 - p_val) / (1 - q_val)
        )

    # Privacy-preserving stopping conditions with Gaussian noise
    theta0 = theta(mu0)
    theta1 = theta(mu1)

    tau0_cond = mean - mu0 + gauss1 <= (
        kl_div(mu0, mu1)
        - jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon_prime)) * beta)) / (n)
    ) / (theta1 - theta0) - gauss2 - jnp.sqrt(
        2
        * (gauss1_std_dev**2 + gauss2_std_dev**2)
        * jnp.log(jnp.maximum(epsilon_prime, 2) * n**s * zeta_s / beta)
    )

    tau1_cond = mean - mu1 + gauss1 >= (
        -kl_div(mu1, mu0)
        + jnp.log(1 / (jnp.maximum(1 / 2, (1 - 1 / epsilon_prime)) * alpha)) / (n)
    ) / (theta1 - theta0) + gauss2 + jnp.sqrt(
        2
        * (gauss1_std_dev**2 + gauss2_std_dev**2)
        * jnp.log(jnp.maximum(epsilon_prime, 2) * n**s * zeta_s / alpha)
    )

    # Compute stopping times
    tau0 = jnp.where(jnp.any(tau0_cond), jnp.argmax(tau0_cond) + 1, len(x))
    tau1 = jnp.where(jnp.any(tau1_cond), jnp.argmax(tau1_cond) + 1, len(x))

    stop_time = jnp.minimum(tau0, tau1)
    decision = jnp.where(stop_time == tau0, -1, 1)
    decision = jnp.where(
        stop_time == len(x), 0, decision
    )  # No decision if max length reached

    # Additional statistics for detailed analysis
    additional_info = {
        "cum_sum": cum_sum,
        "mean": mean,
        "noisy_mean": mean + gauss1,
        "gauss1": gauss1,
        "gauss2": gauss2,
        "gauss3": gauss2,  # Consistent with original lap3 = lap2
        "theta0": theta0,
        "theta1": theta1,
        "tau0_cond": tau0_cond,
        "tau1_cond": tau1_cond,
        "tau0": tau0,
        "tau1": tau1,
        "kl_div_01": kl_div(mu0, mu1),
        "kl_div_10": kl_div(mu1, mu0),
    }

    return decision, stop_time, additional_info
