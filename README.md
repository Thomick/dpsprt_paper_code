# Standalone SPRT Comparison

This package provides a standalone implementation for running Sequential Probability Ratio Test (SPRT) comparison experiments, including differentially private variants. It allows to reproduce results from the paper [Differentially Private Sequential Probability Ratio Tests](https://arxiv.org/abs/2508.06377).

## TL;DR

After installing the dependencies from the `requirements.txt` file, to only reproduce the plots from the main paper, run:

```bash
python main.py \
    --mu0 0.3 \
    --mu1 0.7 \
    --epsilons 0.1 0.5 1.0 2.0 5.0 10.0 25.0 50.0 100.0 \
    --alphas 0.01 0.025 0.05 0.075 0.1 0.15 0.2 \
    --delta 1e-5 \
    --A 1.0 \
    --results-dir results/problem_instances/easy_p0.3_p1.7 \
    --seed 42 \
    --num-trials 1000 \
    --max-samples 10000

python plot_performance_metrics.py \
    --results-file results/problem_instances/easy_p0.3_p1.7/sprt_comparison_results.json \
    --results-dir results/problem_instances/easy_p0.3_p1.7
```

To run all experiments with default parameters, run:

```bash
python run_all.py --seed 42 --max-samples 50000
```
It can be much longer to run all experiments with the default parameters, so we recommend running the experiments with the reduced parameters first to get the results for the plots.

## Overview

Sequential Probability Ratio Tests are statistical methods for hypothesis testing that allow for early stopping once sufficient evidence has been collected. This package implements and compares several SPRT variants:

- **Classical SPRT**: The original algorithm by Wald
- **DP-SPRT**: Differentially private SPRT with privacy-preserving stopping criterion
- **Tuned DP-SPRT**: DP-SPRT with optimized threshold parameters
- **Priv-SPRT**: Privacy-preserving SPRT using truncated log-likelihood ratios and Gaussian noise
- **DP-SPRT (Subsampled)**: DP-SPRT with subsampling
- **DP-SPRT (Gaussian)**: DP-SPRT with Gaussian noise


## Installation

### Requirements

This code requires Python 3.7+ and the following packages:
```
jax==0.6.0
jaxlib==0.6.0
numpy==2.2.5
matplotlib==3.10.3
pandas==2.2.3
tqdm==4.67.1
seaborn==0.13.2
```

To set up the environment:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you have a Nvidia GPU you can enable GPU support by installing
```bash
pip install -U "jax[cuda12]"
```

## Usage

### Running Experiments

To run experiments with default parameters:

```bash
python main.py
```

With custom parameters:

```bash
python main.py --mu0 0.4 --mu1 0.6 --epsilons 0.1 0.5 1.0 2.0 5.0 --alphas 0.05 0.1 0.2 --num-trials 500 --max-samples 5000
```

### Command-line Options

- `--mu0`: Null hypothesis parameter (default: 0.3)
- `--mu1`: Alternative hypothesis parameter (default: 0.7)
- `--epsilons`: List of privacy parameters to test (default: 0.1 0.5 1.0 2.0 5.0)
- `--alphas`: List of alpha=beta values to test (default: 0.05 0.1 0.2)
- `--delta`: Privacy failure probability for PrivSPRT (default: 1e-5)
- `--A`: Truncation parameter for PrivSPRT (default: 1.0)
- `--num-trials`: Number of trials per configuration (default: 500)
- `--max-samples`: Maximum number of samples per sequence (default: 10000)
- `--seed`: Random seed (default: 42)
- `--results-dir`: Directory to save results (default: results)
- `--no-tqdm`: Disable tqdm progress bars
- `--plot-only`: Only generate plots from existing results
- `--results-file`: Path to existing results file (for --plot-only)

### Generating Plots Only

If you have already run experiments and want to generate plots:

```bash
python main.py --plot-only --results-file results/sprt_comparison_results.json
```

### Stopping Time Distributions

To generate plots of stopping time distributions:

```bash
python plot_distributions.py --results-file results/sprt_comparison_results.json
```

## Project Structure

- `main.py`: Main script for running experiments
- `config.py`: Configuration for experiments
- `utils.py`: Utility functions
- `core.py`: Core SPRT algorithm implementations
- `sprt.py`: SPRT algorithm classes
- `experiment.py`: Experiment runner
- `plot_comparison.py`: Functions for plotting comparison results
- `plot_distributions.py`: Functions for plotting stopping time distributions
- `plot_boxplots.py`: Functions for generating boxplot comparisons
- `plot_performance_metrics.py`: Functions for plotting performance metrics
- `plot_dp_sprt_across_instances.py`: Functions for plotting DP-SPRT performance across instances
- `run_all.py`: Script to run all experiments with various configurations

## Reproducibility

To reproduce the experiments from our paper:

```bash
# Run all experiments with the configurations used in the paper
python run_all.py --seed 42

# Generate all plots from the results
python plot_boxplots.py
python plot_distributions.py
python plot_performance_metrics.py
python plot_dp_sprt_across_instances.py
```

## Example Results

The experiments will generate several types of plots:

1. **Performance vs. Epsilon**: Shows how the average sample size and error rates vary with the privacy parameter for each alpha/beta value.

2. **Performance vs. Alpha/Beta**: Shows how performance metrics vary with error rate targets for each epsilon value.

3. **Stopping Time Distributions**: Histograms of stopping times for each algorithm under both the null and alternative hypotheses.

Results are saved to the specified output directory (default: `results/`).

## Customization

To add new SPRT variants or modify existing ones, you can:

1. Add new algorithm implementations in `core.py`
2. Create corresponding algorithm classes in `sprt.py`
3. Add factory functions and update the `create_algorithm_from_config` method in `experiment.py`
4. Update plotting functions in `plot_comparison.py` and `plot_distributions.py`

## Citation

If you use this code in your research, please cite our paper:

```
@article{XYZ2023,
  title={Differentially Private Sequential Probability Ratio Tests},
  author={Author, A. and Author, B.},
  journal={Journal Name},
  year={2023},
  volume={},
  pages={}
}
```

## License

This project is provided for educational and research purposes.

## Contact

For questions or issues, please open an issue on the repository or contact the authors.
