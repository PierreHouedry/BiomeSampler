# README for Simulation Script

## Overview

This Python script provides a comprehensive set of functions and classes to perform synthetic data generation, clustering, residual estimation, and simulation experiments on datasets. The workflow utilizes concepts such as hierarchical clustering, the Bray-Curtis distance, and presence-absence sampling, and it generates synthetic samples based on given data. It also calculates residuals and evaluates the performance of the synthetic samples against the original dataset using various metrics.

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

## Structure

### Utility Functions

1. **is_positive_definite(matrix)**
   - **Input**: `matrix` (ndarray) - The input matrix.
   - **Output**: `bool` - True if the matrix is positive definite, otherwise False.
   - **Description**: Checks if a given matrix is positive definite using Cholesky decomposition.

2. **nearest_positive_definite(matrix)**
   - **Input**: `matrix` (ndarray) - The input matrix.
   - **Output**: `matrix_pd` (ndarray) - A positive definite matrix close to the input.
   - **Description**: Finds the nearest positive definite matrix to the input by adjusting the eigenvalues.

3. **clr_transformation(matrix)**
   - **Input**: `matrix` (ndarray) - The input matrix to transform.
   - **Output**: `clr_matrix` (ndarray) - CLR-transformed matrix.
   - **Description**: Performs a center-log ratio (CLR) transformation on a matrix to deal with compositional data.

4. **silverman_bandwidth(data)**
   - **Input**: `data` (ndarray) - The input data for which the bandwidth is calculated.
   - **Output**: `bandwidth` (float) - The calculated bandwidth.
   - **Description**: Calculates the bandwidth of the input data using Silverman's rule of thumb, often used in kernel density estimation.

### Residual Estimation

1. **estimate_residuals(data_path, n_clusters, result_path, clr_transform=False)**
   - **Input**: 
     - `data_path` (str) - Path to the CSV file containing the data.
     - `n_clusters` (int) - Number of clusters to use for estimating residuals.
     - `result_path` (str) - Path to save the residuals.
     - `clr_transform` (bool, optional) - Whether to apply CLR transformation to the data (default is False).
   - **Output**: `residuals` (DataFrame) - A pandas DataFrame containing the residuals for each observation.
   - **Description**: Estimates residuals by performing hierarchical clustering and fitting a linear regression model. Residuals are saved to the specified result directory.

2. **estimate_n_clusters(data_path, n_min, n_max)**
   - **Input**: 
     - `data_path` (str) - Path to the CSV file containing the data.
     - `n_min` (int) - Minimum number of clusters to evaluate.
     - `n_max` (int) - Maximum number of clusters to evaluate.
   - **Output**: `silhouette_scores` (dict) - A dictionary where keys are the number of clusters and values are the silhouette scores.
   - **Description**: Estimates the optimal number of clusters using silhouette scores, allowing the user to determine the appropriate number of clusters for the data.

### Class Definitions

1. **Sampler**
   - **Description**: Generates synthetic samples based on given bandwidth using uniform noise addition. This class helps to create variability in synthetic data generation.

   - **__init__(data, bandwidth)**
     - **Input**: 
       - `data` (ndarray) - The input data to initialize the sampler.
       - `bandwidth` (float) - The bandwidth value for sampling.
     - **Output**: None
     - **Description**: Initializes the Sampler with data and a bandwidth value.

   - **sample(n_samples)**
     - **Input**: `n_samples` (int) - Number of synthetic samples to generate.
     - **Output**: `sample` (ndarray) - Generated synthetic samples.
     - **Description**: Generates synthetic samples by adding uniform random noise to existing samples.

2. **Simulator**
   - **Description**: Generates synthetic datasets based on a correlation matrix and scores them against the original dataset.

   - **__init__(original_data_path, path)**
     - **Input**: 
       - `original_data_path` (str) - Path to the CSV file containing the original data.
       - `path` (str) - Directory where results will be stored.
     - **Output**: None
     - **Description**: Loads the data, computes a positive definite correlation matrix, and initializes other attributes.

   - **simulate_absence_presence(n_samples)**
     - **Input**: `n_samples` (int) - The number of samples to generate for the absence/presence matrix.
     - **Output**: `absence_presence_matrix` (ndarray) - A binary matrix indicating presence (1) or absence (0) of features.
     - **Description**: Simulates an absence/presence matrix to determine which features are present in each sample.

   - **scoring(sample, display)**
     - **Input**: 
       - `sample` (ndarray) - The generated sample to be compared with the original data.
       - `display` (bool) - If True, prints the computed metrics.
     - **Output**: `metrics` (dict) - A dictionary containing RMSE values for mean, variance, standard deviation, zero proportions, and Bray-Curtis distance.
     - **Description**: Computes scoring metrics such as RMSE for mean, variance, standard deviation, zero proportions, and Bray-Curtis distance.

   - **simulation(n_samples, n_experiments, display=True)**
     - **Input**: 
       - `n_samples` (int) - The number of samples to generate in each experiment.
       - `n_experiments` (int) - The number of simulation experiments to run.
       - `display` (bool, optional) - If True, prints progress and scoring metrics for each experiment (default is True).
     - **Output**: 
       - `experiments` (list of DataFrame) - A list containing generated samples for each experiment.
       - `metrics_list` (list of dict) - A list of dictionaries with metrics for each experiment.
       - `metrics_mean` (dict) - A dictionary with the average metrics across all experiments.
     - **Description**: Runs multiple simulation experiments to generate synthetic samples, save them to the specified folder, and compute metrics for each experiment.





