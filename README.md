## Overview

BiomeSampler is a Python package that provides a comprehensive set of functions and classes to perform synthetic data generation, clustering, residual estimation, and simulation experiments on datasets. The workflow utilizes concepts such as hierarchical clustering, the Bray-Curtis distance, and presence-absence sampling, and it generates synthetic samples based on given data. It also calculates residuals and evaluates the performance of the synthetic samples against the original dataset using various metrics. BiomeSampler is released under the MIT License.

## Installation

You can install BiomeSampler:

```git clone https://github.com/PierreHouedry/BiomeSampler.git
cd BiomeSampler
pip install -r requirements.txt```

## Requirements

The script requires the following libraries:

- NumPy
- Pandas
- SciPy
- scikit-learn
- Pathlib
- datetime

Make sure to install these packages using:

```sh
pip install numpy pandas scipy scikit-learn
```

## Usage

### Importing the Package

```python
from BiomeSampler import Simulator, estimate_n_clusters, estimate_residuals
```

### Simulating Synthetic Data
#### Step 1: Initialize the Simulator

Load your original data from a CSV file where rows represent samples and columns represent features.

```python
original_data_path = 'path/to/your/data.csv'
output_path = 'path/to/save/results'
simulator = Simulator(original_data_path, output_path)
```

#### Step 2: Run the Simulation

Generate synthetic datasets using the simulation method.

```python
n_samples = 100       # Number of synthetic samples to generate
n_experiments = 10    # Number of simulation experiments to run
display = True        # Display progress and metrics
experiments, metrics_list, metrics_mean = simulator.simulation(n_samples, n_experiments, display)
```

#### Step 3: Access the Results

    experiments: List of DataFrames containing synthetic samples for each experiment.
    metrics_list: List of dictionaries with metrics for each experiment.
    metrics_mean: Dictionary with average metrics across all experiments.

### Estimating Optimal Number of Clusters
Determine the optimal number of clusters using silhouette scores.

```python
data_path = 'path/to/your/data.csv'
n_min = 2    # Minimum number of clusters
n_max = 10   # Maximum number of clusters
silhouette_scores = estimate_n_clusters(data_path, n_min, n_max)
```

### Estimating Residuals

Estimate residuals based on Bray-Curtis distances and optional CLR transformation.

```python
data_path = 'path/to/your/data.csv'
n_clusters = 3
result_path = 'path/to/save/results'
clr_transform = False  # Set to True to apply CLR transformation
residuals = estimate_residuals(data_path, n_clusters, result_path, clr_transform)
```

### Data
The data for IBD are available in this repository.

