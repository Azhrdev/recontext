#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Uncertainty quantification module for 3D reconstruction and semantic labeling.

This module provides methods for estimating uncertainty in both geometric 
reconstruction and semantic understanding, enabling more robust integration
of information from multiple sources.

Author: Michael Chen
Date: 2024-02-10
Last modified: 2024-03-08
"""

import numpy as np
import logging
import scipy.stats
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def compute_geometric_uncertainty(points: np.ndarray, 
                                 depths: List[np.ndarray],
                                 confidence: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute geometric uncertainty for 3D points.
    
    Args:
        points: Nx3 array of 3D points
        depths: List of depth values for each point from different views
        confidence: Optional Nx1 array of confidence values for each point
        
    Returns:
        Nx1 array of uncertainty values
    """
    n_points = len(points)
    uncertainty = np.zeros(n_points)
    
    # Compute uncertainty based on depth variance
    for i in range(n_points):
        point_depths = depths[i]
        
        if len(point_depths) <= 1:
            # Not enough observations
            uncertainty[i] = 1.0
            continue
        
        # Compute coefficient of variation (normalized standard deviation)
        mean_depth = np.mean(point_depths)
        if mean_depth > 0:
            std_depth = np.std(point_depths)
            uncertainty[i] = std_depth / mean_depth
        else:
            uncertainty[i] = 1.0
    
    # Scale by confidence if provided
    if confidence is not None:
        uncertainty = uncertainty * (1 - confidence)
    
    # Normalize to [0, 1]
    if np.max(uncertainty) > 0:
        uncertainty = uncertainty / np.max(uncertainty)
    
    return uncertainty


def compute_label_uncertainty(label_candidates: Dict[int, List],
                              point_view_counts: np.ndarray) -> np.ndarray:
    """Compute semantic label uncertainty for 3D points.
    
    Args:
        label_candidates: Dictionary mapping point indices to lists of label candidates
        point_view_counts: Array of view counts for each point
        
    Returns:
        Array of uncertainty values
    """
    # Get maximum point index
    if not label_candidates:
        return np.array([])
        
    max_idx = max(label_candidates.keys())
    uncertainty = np.ones(max_idx + 1)
    
    for point_idx, candidates in label_candidates.items():
        # Skip points with no candidates
        if not candidates:
            continue
        
        # Group by class
        class_votes = defaultdict(float)
        for label in candidates:
            class_id = label.class_id
            score = label.score
            class_votes[class_id] += score
        
        if not class_votes:
            continue
            
        # Normalize votes
        total_votes = sum(class_votes.values())
        if total_votes > 0:
            class_probs = {k: v / total_votes for k, v in class_votes.items()}
        else:
            class_probs = {k: 1.0 / len(class_votes) for k in class_votes}
        
        # Compute entropy
        entropy = compute_entropy(list(class_probs.values()))
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log(max(len(class_probs), 2))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        # Apply view count factor
        view_count = point_view_counts[point_idx]
        if view_count > 0:
            # More views should reduce uncertainty
            view_factor = 1.0 / np.sqrt(view_count)
            uncertainty[point_idx] = normalized_entropy * view_factor
        else:
            uncertainty[point_idx] = 1.0
    
    return uncertainty


def compute_entropy(probabilities: List[float]) -> float:
    """Compute entropy of a probability distribution.
    
    Args:
        probabilities: List of probabilities
        
    Returns:
        Entropy value
    """
    # Filter out zeros and normalize
    probs = np.array([p for p in probabilities if p > 0])
    if len(probs) == 0:
        return 0.0
        
    probs = probs / np.sum(probs)
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs))
    
    return entropy


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    """Normalize probabilities to sum to 1.
    
    Args:
        probs: Array of probabilities
        
    Returns:
        Normalized probabilities
    """
    if np.sum(probs) == 0:
        return np.ones_like(probs) / len(probs)
    else:
        return probs / np.sum(probs)


def estimate_variance_from_samples(samples: np.ndarray) -> np.ndarray:
    """Estimate variance from samples.
    
    Args:
        samples: Array of samples [n_samples, n_dimensions]
        
    Returns:
        Variance for each dimension
    """
    # Check if we have enough samples
    if len(samples) < 2:
        return np.zeros(samples.shape[1])
    
    # Compute variance
    variance = np.var(samples, axis=0)
    
    return variance


def estimate_confidence_interval(samples: np.ndarray, 
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate confidence interval from samples.
    
    Args:
        samples: Array of samples [n_samples, n_dimensions]
        confidence_level: Confidence level
        
    Returns:
        Tuple of (lower bound, upper bound)
    """
    # Check if we have enough samples
    if len(samples) < 2:
        return np.zeros(samples.shape[1]), np.zeros(samples.shape[1])
    
    # Compute mean and standard error
    mean = np.mean(samples, axis=0)
    std_err = scipy.stats.sem(samples, axis=0)
    
    # Compute t-value
    n = len(samples)
    alpha = 1.0 - confidence_level
    t_val = scipy.stats.t.ppf(1 - alpha/2, n-1)
    
    # Compute confidence interval
    lower = mean - t_val * std_err
    upper = mean + t_val * std_err
    
    return lower, upper


def mahalanobis_distance(x: np.ndarray, 
                         mean: np.ndarray, 
                         cov: np.ndarray) -> float:
    """Compute Mahalanobis distance.
    
    Args:
        x: Point
        mean: Mean vector
        cov: Covariance matrix
        
    Returns:
        Mahalanobis distance
    """
    diff = x - mean
    
    # Handle singular covariance matrix
    try:
        inv_cov = np.linalg.inv(cov)
        dist = np.sqrt(diff.dot(inv_cov).dot(diff))
    except np.linalg.LinAlgError:
        # Fallback to Euclidean distance
        dist = np.linalg.norm(diff)
    
    return dist


def bootstrap_estimate_uncertainty(samples: np.ndarray, 
                                  n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate uncertainty using bootstrap resampling.
    
    Args:
        samples: Array of samples [n_samples, n_dimensions]
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Tuple of (mean, standard deviation)
    """
    # Check if we have enough samples
    if len(samples) < 2:
        return np.mean(samples, axis=0), np.zeros(samples.shape[1])
    
    n_samples = len(samples)
    bootstrap_means = np.zeros((n_bootstrap, samples.shape[1]))
    
    # Generate bootstrap samples
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        bootstrap_sample = samples[indices]
        bootstrap_means[i] = np.mean(bootstrap_sample, axis=0)
    
    # Compute mean and standard deviation
    mean = np.mean(bootstrap_means, axis=0)
    std = np.std(bootstrap_means, axis=0)
    
    return mean, std


def bayesian_update_belief(prior_mean: np.ndarray, 
                          prior_var: np.ndarray,
                          obs_mean: np.ndarray,
                          obs_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Update belief using Bayesian update rule.
    
    Args:
        prior_mean: Prior mean
        prior_var: Prior variance
        obs_mean: Observation mean
        obs_var: Observation variance
        
    Returns:
        Tuple of (posterior mean, posterior variance)
    """
    # Handle zero variances
    prior_var = np.maximum(prior_var, 1e-10)
    obs_var = np.maximum(obs_var, 1e-10)
    
    # Compute Kalman gain
    kalman_gain = prior_var / (prior_var + obs_var)
    
    # Update mean and variance
    posterior_mean = prior_mean + kalman_gain * (obs_mean - prior_mean)
    posterior_var = (1 - kalman_gain) * prior_var
    
    return posterior_mean, posterior_var


class UncertaintyModel:
    """Model for tracking and updating uncertainty estimates."""
    
    def __init__(self, dimensions: int):
        """Initialize uncertainty model.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.mean = np.zeros(dimensions)
        self.variance = np.ones(dimensions)
        self.n_observations = 0
    
    def update(self, observation: np.ndarray, 
              observation_variance: Optional[np.ndarray] = None) -> None:
        """Update model with new observation.
        
        Args:
            observation: New observation
            observation_variance: Variance of the observation
        """
        if observation_variance is None:
            # Use default variance that decreases with more observations
            observation_variance = np.ones(self.dimensions) / (self.n_observations + 1)
        
        # Bayesian update
        self.mean, self.variance = bayesian_update_belief(
            self.mean, self.variance, observation, observation_variance)
        
        self.n_observations += 1
    
    def get_confidence_interval(self, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence interval.
        
        Args:
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower bound, upper bound)
        """
        z_value = scipy.stats.norm.ppf((1 + confidence_level) / 2)
        std_dev = np.sqrt(self.variance)
        
        lower = self.mean - z_value * std_dev
        upper = self.mean + z_value * std_dev
        
        return lower, upper
    
    def get_uncertainty(self) -> np.ndarray:
        """Get uncertainty estimate.
        
        Returns:
            Uncertainty values
        """
        # Coefficient of variation
        cv = np.sqrt(self.variance) / (np.abs(self.mean) + 1e-10)
        
        # Scale by number of observations
        uncertainty = cv / np.sqrt(self.n_observations + 1)
        
        return uncertainty


def estimate_label_uncertainty_pointcloud(points: np.ndarray,
                                         labels: np.ndarray,
                                         k_neighbors: int = 10) -> np.ndarray:
    """Estimate label uncertainty based on spatial neighbors.
    
    Args:
        points: Nx3 array of 3D points
        labels: Nx1 array of labels
        k_neighbors: Number of neighbors to consider
        
    Returns:
        Nx1 array of uncertainty values
    """
    try:
        from sklearn.neighbors import NearestNeighbors
        
        # Build neighbor model
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 to include the point itself
        nn.fit(points)
        
        # Find neighbors
        _, indices = nn.kneighbors(points)
        
        # Compute label entropy for each point's neighborhood
        n_points = len(points)
        uncertainty = np.zeros(n_points)
        
        for i in range(n_points):
            # Get labels of neighbors
            neighbor_indices = indices[i][1:]  # Skip the point itself
            neighbor_labels = labels[neighbor_indices]
            
            # Count label occurrences
            label_counts = defaultdict(int)
            for label in neighbor_labels:
                if label >= 0:  # Ignore unlabeled points
                    label_counts[label] += 1
            
            # Compute probabilities
            total = sum(label_counts.values())
            if total > 0:
                probs = [count / total for count in label_counts.values()]
                
                # Compute entropy
                entropy = compute_entropy(probs)
                
                # Normalize by maximum possible entropy
                max_entropy = np.log(max(len(label_counts), 2))
                if max_entropy > 0:
                    uncertainty[i] = entropy / max_entropy
                else:
                    uncertainty[i] = 0.0
            else:
                uncertainty[i] = 1.0  # Maximum uncertainty if all neighbors are unlabeled
        
        return uncertainty
        
    except ImportError:
        logger.warning("scikit-learn not available, using simplified uncertainty estimation")
        
        # Simplified version: uncertainty is 1 if unlabeled, 0 otherwise
        return np.array([1.0 if label < 0 else 0.0 for label in labels])


def propagate_uncertainty(means: np.ndarray, 
                         variances: np.ndarray,
                         function: callable,
                         method: str = 'linearization') -> Tuple[np.ndarray, np.ndarray]:
    """Propagate uncertainty through a function.
    
    Args:
        means: Mean values
        variances: Variance values
        function: Function to propagate through
        method: Method for propagation ('linearization' or 'monte_carlo')
        
    Returns:
        Tuple of (output mean, output variance)
    """
    if method == 'linearization':
        # Linearization method (first-order Taylor approximation)
        output_mean = function(means)
        
        # Estimate gradient numerically
        eps = 1e-6
        gradient = np.zeros_like(means)
        
        for i in range(len(means)):
            means_plus = means.copy()
            means_plus[i] += eps
            
            gradient[i] = (function(means_plus) - output_mean) / eps
        
        # Propagate variance
        output_variance = np.sum(gradient**2 * variances)
        
        return output_mean, output_variance
        
    elif method == 'monte_carlo':
        # Monte Carlo method
        n_samples = 1000
        
        # Generate samples
        samples = np.random.normal(means, np.sqrt(variances), (n_samples, len(means)))
        
        # Propagate samples
        outputs = np.array([function(sample) for sample in samples])
        
        # Compute statistics
        output_mean = np.mean(outputs)
        output_variance = np.var(outputs)
        
        return output_mean, output_variance
        
    else:
        raise ValueError(f"Unknown uncertainty propagation method: {method}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test entropy calculation
    print("Testing entropy calculation:")
    
    # Uniform distribution
    uniform_probs = [0.25, 0.25, 0.25, 0.25]
    uniform_entropy = compute_entropy(uniform_probs)
    print(f"Entropy of uniform distribution: {uniform_entropy:.4f}")
    
    # Certain distribution
    certain_probs = [1.0, 0.0, 0.0, 0.0]
    certain_entropy = compute_entropy(certain_probs)
    print(f"Entropy of certain distribution: {certain_entropy:.4f}")
    
    # Skewed distribution
    skewed_probs = [0.7, 0.1, 0.1, 0.1]
    skewed_entropy = compute_entropy(skewed_probs)
    print(f"Entropy of skewed distribution: {skewed_entropy:.4f}")
    
    # Test uncertainty model
    print("\nTesting uncertainty model:")
    
    model = UncertaintyModel(dimensions=2)
    print(f"Initial mean: {model.mean}")
    print(f"Initial variance: {model.variance}")
    
    # Add observations
    observations = [
        np.array([1.0, 2.0]),
        np.array([1.2, 1.8]),
        np.array([0.9, 2.1]),
        np.array([1.1, 1.9])
    ]
    
    for i, obs in enumerate(observations):
        model.update(obs)
        print(f"After observation {i+1}:")
        print(f"  Mean: {model.mean}")
        print(f"  Variance: {model.variance}")
        print(f"  Uncertainty: {model.get_uncertainty()}")
        
    # Test confidence interval
    lower, upper = model.get_confidence_interval(0.95)
    print(f"\n95% confidence interval:")
    print(f"  Lower: {lower}")
    print(f"  Upper: {upper}")
    
    # Test bootstrap uncertainty estimation
    print("\nTesting bootstrap uncertainty estimation:")
    
    # Generate some data
    true_mean = np.array([5.0, 10.0])
    true_std = np.array([1.0, 2.0])
    samples = np.random.normal(true_mean, true_std, (20, 2))
    
    # Estimate using bootstrap
    boot_mean, boot_std = bootstrap_estimate_uncertainty(samples, n_bootstrap=1000)
    
    print(f"True mean: {true_mean}, Bootstrap mean: {boot_mean}")
    print(f"True std: {true_std}, Bootstrap std: {boot_std}")
    
    # Test Bayesian update
    print("\nTesting Bayesian update:")
    
    prior_mean = np.array([0.0, 0.0])
    prior_var = np.array([1.0, 1.0])
    
    obs_mean = np.array([1.0, 2.0])
    obs_var = np.array([0.5, 0.5])
    
    post_mean, post_var = bayesian_update_belief(prior_mean, prior_var, obs_mean, obs_var)
    
    print(f"Prior mean: {prior_mean}, Prior variance: {prior_var}")
    print(f"Observation mean: {obs_mean}, Observation variance: {obs_var}")
    print(f"Posterior mean: {post_mean}, Posterior variance: {post_var}")