import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh
from scipy.stats import multivariate_normal, norm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import iqr
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
from datetime import datetime

# ==============================================================================
# Utility Functions
# ==============================================================================

def is_positive_definite(matrix):
    """
    Check if a matrix is positive definite.

    Parameters:
    - matrix (ndarray): The input matrix.
    """
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def nearest_positive_definite(matrix):
    """
    Find the nearest positive definite matrix to the input.

    Parameters:
    - matrix (ndarray): The input matrix.

    Returns:
    - matrix_pd (ndarray): A positive definite matrix close to the input.
    """
    B = (matrix + matrix.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    matrix_pd = (B + H) / 2
    matrix_pd = (matrix_pd + matrix_pd.T) / 2

    if is_positive_definite(matrix_pd):
        return matrix_pd

    spacing = np.spacing(np.linalg.norm(matrix))
    I = np.eye(matrix.shape[0])
    k = 1
    while not is_positive_definite(matrix_pd):
        mineig = np.min(np.real(eigh(matrix_pd)[0]))
        adjustment = I * (-mineig * k**2 + spacing)
        matrix_pd += adjustment
        k += 1

    matrix_pd = (matrix_pd + matrix_pd.T) / 2

    return matrix_pd


def clr_transformation(matrix):
    """
    Center-log ratio (CLR) transformation of a matrix.

    Parameters:
    - matrix (ndarray): The input matrix to transform.

    Returns:
    - clr_matrix (ndarray): CLR-transformed matrix.
    """

    matrix = np.array(matrix)

    matrix[matrix==0]=1e-6

    geometric_mean = np.exp(np.mean(np.log(matrix), axis=1))
    
    # Apply CLR transformation
    clr_matrix = np.log(matrix / geometric_mean[:, np.newaxis])
    
    return clr_matrix

def silverman_bandwidth(data):
    """
    Calculate bandwidth using Silverman's rule of thumb.

    Parameters:
    - data (ndarray): The input data for which the bandwidth is calculated.

    Returns:
    - bandwidth (float): The calculated bandwidth.
    """
    n = len(data)
    std_dev = np.std(data)
    iqr_val = iqr(data)
    bandwidth = 0.9 * min(std_dev, iqr_val / 1.34) * n**(-1/5)

    return bandwidth

def estimate_residuals(data_path, n_clusters, result_path, clr_transform=False):
        """
        Estimate residuals based on Bray-Curtis distances and CLR-transformed data.

        Parameters:
        - n_clusters (int): Number of clusters to use for estimating residuals.

        Returns:
        - residuals (DataFrame): A pandas DataFrame containing the residuals for each observation.
        """

        data = pd.read_csv(data_path, index_col=0).to_numpy()
        result_path = Path(result_path)
        
        # Compute Bray-Curtis distance matrix (condensed format)
        bray_curtis = pdist(data, metric='braycurtis')
        
        if clr_transform:
            # Apply CLR transformation to the data
            data = clr_transformation(data)
        
        # Perform hierarchical clustering using Ward's method
        Z = linkage(bray_curtis, method='ward')
        
        # Form 3 clusters from the linkage matrix
        Z_1 = fcluster(Z, n_clusters, criterion='maxclust')

        # Encode cluster labels into a DataFrame
        Z_1_encoded = pd.DataFrame(Z_1, columns=['Cluster'])  
        
        # Fit linear model to clusters and CLR-transformed data
        model = LinearRegression()
        model.fit(Z_1_encoded, data)
        
        # Compute residuals from the model's predictions
        residuals = data - model.predict(Z_1_encoded)

        residuals = pd.DataFrame(residuals)
        residual_dir = result_path/'Residuals'
        residual_dir.mkdir(exist_ok=True)
        residuals.to_csv(residual_dir/f'residuals_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

        return residuals    

    
def estimate_n_clusters(data_path, n_min, n_max):
        """
        Estimate the optimal number of clusters using silhouette scores.

        Parameters:
        - n_min (int): Minimum number of clusters to evaluate.
        - n_max (int): Maximum number of clusters to evaluate.

        Returns:
        - silhouette_scores (dict): A dictionary where keys are the number of clusters and values are the silhouette scores.
        """

        data = pd.read_csv(data_path, index_col=0).to_numpy()
          
        silhouette_scores = {}

        # Compute condensed Bray-Curtis distance matrix for clustering
        condensed_distance_matrix = pdist(data, metric='braycurtis')

        # Perform hierarchical clustering using Ward's method
        Z = linkage(condensed_distance_matrix, method='ward')

        # Convert condensed distance matrix to a square form for silhouette scoring
        square_distance_matrix = squareform(condensed_distance_matrix)
        
        print("\nEstimating optimal number of clusters:\n" + "-"*40)
        for n_clusters in range(n_min, n_max + 1):

            # Form clusters from linkage matrix based on current cluster count
            cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
            
            # Calculate silhouette score for the current number of clusters
            score = silhouette_score(square_distance_matrix, cluster_labels, metric='precomputed')
            silhouette_scores[n_clusters] = score
            print(f"Number of clusters: {n_clusters:2d} | Silhouette score: {score:.4f}")
        
        print("-" * 40)

        optimal_clusters = max(silhouette_scores, key=silhouette_scores.get)
        optimal_score = silhouette_scores[optimal_clusters]
        
        print(f"Optimal number of clusters: {optimal_clusters} with silhouette score: {optimal_score:.4f}")
        
        return silhouette_scores


# ==============================================================================
# Class Definition
# ==============================================================================

class Sampler():
    """
    Sampler class to generate synthetic samples based on bandwidth.
    """

    def __init__(self, data, bandwidth):
        self.data = data
        self.bandwidth = bandwidth

    def sample(self, n_samples): 
        indices = np.random.choice(len(self.data), size=n_samples, replace=True)
        random_samples = self.data[indices]
        u = np.random.uniform(low=-1, high=1, size=random_samples.shape)
        sample = random_samples + self.bandwidth * u
        return sample

class Simulator():
    """
    Sampler class to generate synthetic samples based on a given bandwidth.
    """

    def __init__(self, original_data_path, path):
        """
        Initialize the Simulator with data, path, and optional columns to remove.

        Parameters:
        - original_data_path (str): Path to the CSV file containing the data.
        - path (str): Directory where results will be stored.

        Attributes:
        - original_data (ndarray): The original data loaded from the CSV file.
        - data (ndarray): The processed data after removing specified and zero standard deviation columns.
        - correlation_matrix (ndarray): Correlation matrix of the processed data, corrected to be positive definite.
        - zero_proportions (ndarray): Proportion of zeros in each column of the processed data.
        """

        self.path = Path(path)
        self.path.mkdir(exist_ok=True, parents=True)
        self.original_data = pd.read_csv(original_data_path, index_col=0).to_numpy()

        std_dev = np.std(self.original_data, axis=0)
        columns_with_zero_std = np.where(std_dev == 0)[0]
        if len(columns_with_zero_std) > 0:
            print(f"Removing columns with zero standard deviation: {columns_with_zero_std}")
        self.data = np.delete(self.original_data, columns_with_zero_std, axis=1)
        self.correlation_matrix = nearest_positive_definite(
            np.corrcoef((self.data != 0).astype(int), rowvar=False)
        )
        self.zero_proportions = np.mean(((self.data != 0) == 0), axis=0)


    def simulate_absence_presence(self, n_samples):
        """
        Simulate an absence/presence matrix based on the correlation matrix and zero proportions.

        Parameters:
        - n_samples (int): The number of samples to generate for the absence/presence matrix.

        Returns:
        - absence_presence_matrix (ndarray): A binary matrix of shape (n_samples, n_features), 
          where 1 indicates presence and 0 indicates absence of a feature.
        """
        n_features = len(self.zero_proportions)

        # Create a multivariate normal distribution based on the correlation matrix
        mvn = multivariate_normal(
            mean=np.zeros(n_features), 
            cov=self.correlation_matrix, 
            allow_singular=True
        )
        # Generate correlated normal samples for each feature
        correlated_normals = mvn.rvs(size=n_samples)

        # Calculate thresholds based on zero proportions using the inverse normal distribution
        thresholds = norm.ppf(self.zero_proportions)

        # Simulate presence (1) or absence (0) for each feature by comparing against thresholds
        absence_presence_matrix = (correlated_normals > thresholds).astype(int)

        return absence_presence_matrix
    
    def scoring(self, sample, display):
        """
        Compute scoring metrics between the original data and a generated sample.

        Parameters:
        - sample (ndarray): The generated sample to be compared with the original data.
        - display (bool): If True, print the computed metrics.

        Returns:
        - metrics (dict): A dictionary containing RMSE values for mean, variance, 
          standard deviation, zero proportions, and Bray-Curtis distance between 
          the original and sampled data.
        """
        metrics = {}
    
        # Calculate RMSE for mean, variance, standard deviation, and zero proportions
        metrics['RMSE_mean'] = root_mean_squared_error(
            self.data.mean(axis=0), sample.mean(axis=0)
        )
        metrics['RMSE_var'] = root_mean_squared_error(
            self.data.var(axis=0), sample.var(axis=0)
        )
        metrics['RMSE_std'] = root_mean_squared_error(
            self.data.std(axis=0, ddof=1), sample.std(axis=0, ddof=1)
        )
        metrics['RMSE_zero_proportions'] = root_mean_squared_error(
            self.zero_proportions, (sample == 0).mean(axis=0) 
        )
    
        # Remove columns that are all zeros in the sample
        zero_columns_indices = np.where(np.all(sample == 0, axis=0))[0]   
        sample_dropped = np.delete(sample, zero_columns_indices, axis=1)
        data_dropped = np.delete(self.data, zero_columns_indices, axis=1) 

        # Compute Bray-Curtis distance matrices
        bray_curtis_original = squareform(pdist(data_dropped.T, metric='braycurtis'))
        bray_curtis_sampled = squareform(pdist(sample_dropped.T, metric='braycurtis'))
        metrics['RMSE_Bray_curtis'] = root_mean_squared_error(
            bray_curtis_sampled, bray_curtis_original 
        )

        # Print metrics
        if display:
            print("\nScoring Metrics:\n" + "-"*30)
            for metric, value in metrics.items():
                print(f"{metric:<25}: {value:.6f}")
            print("-" * 30)

        return metrics


    def simulation(self, n_samples, n_experiments, display):
        """
        Run simulations to generate synthetic datasets and compute metrics.

        Parameters:
        - n_samples (int): The number of samples to generate in each experiment.
        - n_experiments (int): The number of simulation experiments to run.
        - display (bool): If True, print progress and scoring metrics for each experiment.

        Returns:
        - experiments (list of DataFrame): A list containing generated samples for each experiment.
        - metrics_list (list of dict): A list of dictionaries with metrics for each experiment.
        - metrics_mean (dict): A dictionary with the average metrics across all experiments.
        """
        
        experiments = []
        metrics_list = []
        for k in range(n_experiments):
            
            if display:
                print(f"\nRunning sampling {k+1}/{n_experiments}")

            # Initialize samplers for each feature based on non-zero data
            presence_absence_data = self.simulate_absence_presence(n_samples)
            n_features = self.data.shape[1]
            models = []
            
            sample_dir = self.path/'Sampling'
            sample_dir.mkdir(exist_ok=True)

            # Initialize Samplers for each feature
            for i in range(n_features):
                non_zero_data = self.data[:, i][self.data[:, i] > 0]

                # If no non-zero data exists, append None as a placeholder
                if len(non_zero_data) == 0:
                    models.append(None)
                    continue
                best_bandwidth = silverman_bandwidth(non_zero_data)

                # Initialize the sampler for the feature
                models.append(Sampler(data=non_zero_data.reshape(-1,1), bandwidth=best_bandwidth))
            
            # Generate synthetic samples based on presence/absence and samplers
            sample = np.zeros((n_samples, n_features))
            for j in range(n_samples):
                for i, model in enumerate(models):
                    if model is not None and presence_absence_data[j, i] == 1:
                        sample[j, i] = model.sample(1)[0][0]
            sample[sample<0] = 0
            sample_csv = pd.DataFrame(sample)
            sample_csv.to_csv(sample_dir/f'samples_experiments_{k}.csv')
            experiments.append(sample_csv)
            metrics = self.scoring(sample, display)
            metrics_list.append(metrics)
            
        metrics_mean = {}
        for key in metrics_list[0]:
            values = [d[key] for d in metrics_list]
            metrics_mean[key] = np.mean(values)

        if display:
            print("\nAverage Scoring Metrics on all simulations\n" + "-"*30)
            for metric, value in metrics_mean.items():
                print(f"{metric:<25}: {value:.10}")
            print("-" * 30)
            
        return experiments, metrics_list, metrics_mean