Synthetic Data Simulator

A Python package for simulating synthetic datasets based on presence/absence matrices derived from a correlation matrix. The simulator allows for generating data, performing clustering, estimating residuals, and computing scoring metrics to compare the synthetic data with the original dataset.


# Installation

To install the package, clone this repository and install the required dependencies:

```bash
git clone <repo-url>
cd synthetic-data-simulator
pip install -r requirements.txt
```

# Usage

The primary class in this package is Simulator, which helps you create and evaluate synthetic datasets. To get started, create an instance of the Simulator class with your data file and output path.

```python
from synthetic_data_simulator import Simulator

# Initialize the simulator with your data
sim = Simulator(original_data_path='data.csv', path='results')

# Run simulations and generate synthetic datasets
experiments, metrics_list, metrics_mean = sim.simulation(
    n_samples=100, 
    n_experiments=10, 
    display=True
)
```

# Classes and Methods
## Simulator Class

The Simulator class is responsible for generating synthetic datasets, performing clustering, simulating presence/absence matrices, and computing residuals and scoring metrics.

## Initialization

Parameters:

    original_data_path (str): Path to the CSV file containing the data.
    path (str): Directory where results will be stored.
    columns_to_remove (list of int, optional): List of column indices to remove from the dataset. If None, no columns are removed initially based on user input.


```python
sim = Simulator(original_data_path, path, columns_to_remove=None)
```

## simulate_absence_presence

Simulates a presence/absence matrix based on the correlation matrix and the proportions of zeros in each column.

Parameters:

    n_samples (int): The number of samples to generate for the absence/presence matrix.

Returns:

    absence_presence_matrix (ndarray): A binary matrix of shape (n_samples, n_features), where 1 indicates presence and 0 indicates absence of a feature.

```python

absence_presence_matrix = sim.simulate_absence_presence(n_samples)
```

## scoring

Calculates scoring metrics between the original data and a generated sample.

Parameters:

    sample (ndarray): The generated sample to be compared with the original data.
    display (bool): If True, print the computed metrics.

Returns:

    metrics (dict): A dictionary containing RMSE values for mean, variance, standard deviation, zero proportions, and Bray-Curtis distance between the original and sampled data.

```python
metrics = sim.scoring(sample, display)
```

## simulation

Generates synthetic datasets and computes metrics to evaluate their similarity to the original data.

Parameters:

    n_samples (int): The number of samples to generate in each experiment.
    n_experiments (int): The number of simulation experiments to run.
    display (bool): If True, print progress and scoring metrics for each experiment.

Returns:

    experiments (list of DataFrame): A list containing generated samples for each experiment.
    metrics_list (list of dict): A list of dictionaries with metrics for each experiment.
    metrics_mean (dict): A dictionary with the average metrics across all experiments.

```python
experiments, metrics_list, metrics_mean = sim.simulation(n_samples, n_experiments, display)
```


## estimate_n_clusters

Estimates the optimal number of clusters using silhouette scores.

Parameters:

    n_min (int): Minimum number of clusters to evaluate.
    n_max (int): Maximum number of clusters to evaluate.

Returns:

    silhouette_scores (dict): A dictionary where keys are the number of clusters and values are the silhouette scores.

```python
silhouette_scores = sim.estimate_n_clusters(n_min, n_max)
```

## estimate_residuals

Estimates residuals based on Bray-Curtis distances and CLR-transformed data.

Parameters:

    n_clusters (int): Number of clusters to use for estimating residuals.

Returns:

    residuals (DataFrame): A pandas DataFrame containing the residuals for each observation.

```python
residuals = sim.estimate_residuals(n_clusters)
```

# Contributing

Contributions are welcome! Please fork the repository and create a pull request for any features or bug fixes.
License

This project is licensed under the MIT License - see the LICENSE file for details.
