#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consensus algorithms for integrating semantic segmentation from multiple views.

This module implements various consensus strategies to resolve conflicts when
projecting 2D semantic labels onto 3D geometry from multiple camera views.

Author: James Chen
Date: 2024-01-30
Last modified: 2024-03-15
"""

import numpy as np
import torch
import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
from scipy import stats

from recontext.integration.uncertainty import compute_entropy, normalize_probabilities

logger = logging.getLogger(__name__)

class ConsensusStrategy:
    """Base class for consensus strategies."""
    
    def __init__(self, name: str):
        """Initialize consensus strategy.
        
        Args:
            name: Strategy name
        """
        self.name = name
    
    def resolve(self, 
                candidates: List[Tuple[int, float]], 
                weights: Optional[List[float]] = None) -> Tuple[int, float]:
        """Resolve conflicts among label candidates.
        
        Args:
            candidates: List of (label_id, confidence) tuples
            weights: Optional list of weights for each candidate
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        raise NotImplementedError("Subclasses must implement resolve")


class MajorityVotingConsensus(ConsensusStrategy):
    """Simple majority voting consensus strategy."""
    
    def __init__(self):
        """Initialize majority voting consensus."""
        super().__init__("majority_voting")
    
    def resolve(self, 
                candidates: List[Tuple[int, float]], 
                weights: Optional[List[float]] = None) -> Tuple[int, float]:
        """Resolve using majority voting.
        
        Args:
            candidates: List of (label_id, confidence) tuples
            weights: Optional list of weights for each candidate
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        if not candidates:
            return (-1, 0.0)  # No candidates
            
        if len(candidates) == 1:
            return candidates[0]  # Only one candidate
        
        # Count votes for each label
        label_counts = Counter([label for label, _ in candidates])
        
        # Find label with most votes
        most_common = label_counts.most_common(1)[0]
        winner_label = most_common[0]
        vote_count = most_common[1]
        
        # Calculate confidence as ratio of votes for winning label
        confidence = vote_count / len(candidates)
        
        # Improve confidence by considering original confidences
        confidences = [conf for label, conf in candidates if label == winner_label]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Combine vote ratio and average confidence
        final_confidence = 0.7 * confidence + 0.3 * avg_confidence
        
        return (winner_label, final_confidence)


class WeightedVotingConsensus(ConsensusStrategy):
    """Weighted voting consensus strategy."""
    
    def __init__(self):
        """Initialize weighted voting consensus."""
        super().__init__("weighted_voting")
    
    def resolve(self, 
                candidates: List[Tuple[int, float]], 
                weights: Optional[List[float]] = None) -> Tuple[int, float]:
        """Resolve using weighted voting.
        
        Args:
            candidates: List of (label_id, confidence) tuples
            weights: Optional list of weights for each candidate
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        if not candidates:
            return (-1, 0.0)  # No candidates
            
        if len(candidates) == 1:
            return candidates[0]  # Only one candidate
        
        # Use default weights if not provided
        if weights is None:
            weights = [1.0] * len(candidates)
        
        # Use confidence as additional weighting factor
        combined_weights = [w * c for (_, c), w in zip(candidates, weights)]
        
        # Count weighted votes for each label
        label_votes = defaultdict(float)
        for (label, _), weight in zip(candidates, combined_weights):
            label_votes[label] += weight
        
        # Find label with highest weighted votes
        winner_label = max(label_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate confidence as ratio of weighted votes for winning label
        total_weight = sum(combined_weights)
        confidence = label_votes[winner_label] / total_weight if total_weight > 0 else 0.0
        
        return (winner_label, min(confidence, 1.0))


class BayesianConsensus(ConsensusStrategy):
    """Bayesian consensus strategy for combining evidence."""
    
    def __init__(self, prior_strength: float = 0.1):
        """Initialize Bayesian consensus.
        
        Args:
            prior_strength: Strength of prior (higher values give more weight to prior)
        """
        super().__init__("bayesian")
        self.prior_strength = prior_strength
    
    def resolve(self, 
                candidates: List[Tuple[int, float]], 
                weights: Optional[List[float]] = None) -> Tuple[int, float]:
        """Resolve using Bayesian consensus.
        
        Args:
            candidates: List of (label_id, confidence) tuples
            weights: Optional list of weights for each candidate
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        if not candidates:
            return (-1, 0.0)  # No candidates
            
        if len(candidates) == 1:
            return candidates[0]  # Only one candidate
        
        # Use default weights if not provided
        if weights is None:
            weights = [1.0] * len(candidates)
        
        # Get unique labels
        unique_labels = set(label for label, _ in candidates)
        
        # Create uniform prior
        num_labels = len(unique_labels)
        prior = {label: self.prior_strength / num_labels for label in unique_labels}
        
        # Initialize posterior with prior
        posterior = prior.copy()
        
        # Update posterior with each observation
        for (label, conf), weight in zip(candidates, weights):
            # Skip low-confidence observations
            if conf < 0.1:
                continue
                
            # Convert confidence to probability
            prob = conf * weight
            
            # Update posterior for this label
            if label in posterior:
                posterior[label] *= (1.0 + prob)
            
            # Update posterior for other labels (reduces their probability)
            for other_label in posterior:
                if other_label != label:
                    posterior[other_label] *= (1.0 - prob / (num_labels - 1))
        
        # Normalize posterior
        total = sum(posterior.values())
        if total > 0:
            posterior = {k: v / total for k, v in posterior.items()}
        
        # Select label with highest posterior probability
        winner_label = max(posterior.items(), key=lambda x: x[1])[0]
        confidence = posterior[winner_label]
        
        return (winner_label, confidence)


class ConfidenceWeightedConsensus(ConsensusStrategy):
    """Confidence-weighted consensus strategy."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize confidence-weighted consensus.
        
        Args:
            confidence_threshold: Minimum confidence threshold
        """
        super().__init__("confidence_weighted")
        self.confidence_threshold = confidence_threshold
    
    def resolve(self, 
                candidates: List[Tuple[int, float]], 
                weights: Optional[List[float]] = None) -> Tuple[int, float]:
        """Resolve using confidence-weighted consensus.
        
        Args:
            candidates: List of (label_id, confidence) tuples
            weights: Optional list of weights for each candidate
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        if not candidates:
            return (-1, 0.0)  # No candidates
            
        if len(candidates) == 1:
            return candidates[0]  # Only one candidate
        
        # Filter out low-confidence candidates
        filtered_candidates = [(label, conf) for label, conf in candidates 
                              if conf >= self.confidence_threshold]
        
        # If all candidates are low-confidence, use the original list
        if not filtered_candidates:
            filtered_candidates = candidates
        
        # Sort by confidence (descending)
        sorted_candidates = sorted(filtered_candidates, key=lambda x: x[1], reverse=True)
        
        # Take the highest confidence candidate
        winner_label, confidence = sorted_candidates[0]
        
        # Check if multiple candidates have similar high confidence
        similar_conf_candidates = [(label, conf) for label, conf in sorted_candidates 
                                  if abs(conf - confidence) < 0.1]
        
        if len(similar_conf_candidates) > 1:
            # Multiple candidates with similar confidence, use majority vote
            majority = Counter([label for label, _ in similar_conf_candidates]).most_common(1)[0]
            winner_label = majority[0]
            
            # Adjust confidence based on agreement
            agreement_ratio = majority[1] / len(similar_conf_candidates)
            confidence = confidence * agreement_ratio
        
        return (winner_label, confidence)


class EnsembleConsensus(ConsensusStrategy):
    """Ensemble consensus strategy that combines multiple consensus methods."""
    
    def __init__(self):
        """Initialize ensemble consensus."""
        super().__init__("ensemble")
        
        # Create component consensus strategies
        self.strategies = [
            MajorityVotingConsensus(),
            WeightedVotingConsensus(),
            BayesianConsensus(),
            ConfidenceWeightedConsensus()
        ]
    
    def resolve(self, 
                candidates: List[Tuple[int, float]], 
                weights: Optional[List[float]] = None) -> Tuple[int, float]:
        """Resolve using ensemble of consensus strategies.
        
        Args:
            candidates: List of (label_id, confidence) tuples
            weights: Optional list of weights for each candidate
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        if not candidates:
            return (-1, 0.0)  # No candidates
            
        if len(candidates) == 1:
            return candidates[0]  # Only one candidate
        
        # Apply each strategy
        results = []
        for strategy in self.strategies:
            label, conf = strategy.resolve(candidates, weights)
            results.append((label, conf))
        
        # Count votes for each label
        label_counts = Counter([label for label, _ in results])
        
        # Find label with most votes
        winner_label = label_counts.most_common(1)[0][0]
        
        # Calculate average confidence for the winning label
        confidences = [conf for label, conf in results if label == winner_label]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return (winner_label, avg_confidence)


def create_consensus_strategy(strategy_name: str, **kwargs) -> ConsensusStrategy:
    """Factory function for creating consensus strategies.
    
    Args:
        strategy_name: Name of strategy to create
        **kwargs: Additional parameters for the strategy
        
    Returns:
        ConsensusStrategy instance
    """
    strategies = {
        'majority': MajorityVotingConsensus,
        'weighted': WeightedVotingConsensus,
        'bayesian': BayesianConsensus,
        'confidence': ConfidenceWeightedConsensus,
        'ensemble': EnsembleConsensus
    }
    
    if strategy_name not in strategies:
        logger.warning(f"Unknown consensus strategy: {strategy_name}, using majority voting")
        return MajorityVotingConsensus()
    
    return strategies[strategy_name](**kwargs)


def apply_consensus(label_candidates: Dict[int, List[Tuple[int, float, float]]],
                   strategy_name: str = 'ensemble',
                   **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Apply consensus to resolve label conflicts.
    
    Args:
        label_candidates: Dictionary mapping point indices to list of 
                         (label_id, confidence, weight) tuples
        strategy_name: Name of consensus strategy to use
        **kwargs: Additional parameters for consensus strategy
        
    Returns:
        Tuple of (label_array, confidence_array)
    """
    if not label_candidates:
        return np.array([]), np.array([])
    
    # Create consensus strategy
    strategy = create_consensus_strategy(strategy_name, **kwargs)
    logger.info(f"Using consensus strategy: {strategy.name}")
    
    # Get maximum point index
    max_idx = max(label_candidates.keys())
    
    # Initialize output arrays
    labels = np.full(max_idx + 1, -1, dtype=np.int32)
    confidences = np.zeros(max_idx + 1, dtype=np.float32)
    
    # Process each point
    num_resolved = 0
    for point_idx, candidates in label_candidates.items():
        # Skip points with no candidates
        if not candidates:
            continue
        
        # Extract labels, confidences, and weights
        label_confs = [(label, conf) for label, conf, _ in candidates]
        weights = [weight for _, _, weight in candidates]
        
        # Apply consensus strategy
        label, conf = strategy.resolve(label_confs, weights)
        
        # Store result
        labels[point_idx] = label
        confidences[point_idx] = conf
        
        if label >= 0:
            num_resolved += 1
    
    logger.info(f"Resolved labels for {num_resolved}/{len(label_candidates)} points")
    
    return labels, confidences


def smoothen_labels(points: np.ndarray,
                   labels: np.ndarray,
                   confidences: np.ndarray,
                   radius: float = 0.05,
                   confidence_threshold: float = 0.3) -> np.ndarray:
    """Smoothen labels using spatial consistency.
    
    Args:
        points: Point coordinates (N x 3)
        labels: Point labels (N)
        confidences: Label confidences (N)
        radius: Neighborhood radius
        confidence_threshold: Minimum confidence for neighbors to influence
        
    Returns:
        Smoothened labels
    """
    logger.info(f"Smoothening labels with radius {radius}")
    
    # Skip if too few points
    if len(points) < 10:
        return labels
    
    try:
        # Try to use Open3D for efficient neighbor search
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Create KD-tree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        # New labels
        new_labels = labels.copy()
        
        # For each point
        for i in range(len(points)):
            # Skip points with high confidence
            if confidences[i] > 0.9:
                continue
                
            # Skip unlabeled points
            if labels[i] < 0:
                continue
                
            # Find neighbors within radius
            _, indices, _ = kdtree.search_radius_vector_3d(points[i], radius)
            
            # Skip self
            indices = [idx for idx in indices if idx != i]
            
            # Skip if no neighbors
            if not indices:
                continue
                
            # Filter neighbors by confidence
            neighbors = [(labels[idx], confidences[idx]) for idx in indices
                        if confidences[idx] >= confidence_threshold]
            
            # Skip if no confident neighbors
            if not neighbors:
                continue
                
            # Apply weighted voting
            strategy = WeightedVotingConsensus()
            label, conf = strategy.resolve(neighbors)
            
            # Update label if majority is different and confidence is higher
            if label != labels[i] and conf > confidences[i]:
                new_labels[i] = label
        
        changed = np.sum(new_labels != labels)
        logger.info(f"Changed {changed} labels during smoothening")
        
        return new_labels
        
    except ImportError:
        logger.warning("Open3D not available for efficient smoothening")
        return labels


def compute_label_consistency(points: np.ndarray,
                             labels: np.ndarray,
                             radius: float = 0.1) -> np.ndarray:
    """Compute label consistency score for each point.
    
    Args:
        points: Point coordinates (N x 3)
        labels: Point labels (N)
        radius: Neighborhood radius
        
    Returns:
        Consistency scores (N)
    """
    logger.info(f"Computing label consistency with radius {radius}")
    
    # Skip if too few points
    if len(points) < 10:
        return np.ones(len(points))
    
    try:
        # Try to use Open3D for efficient neighbor search
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Create KD-tree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        # Consistency scores
        consistency = np.zeros(len(points))
        
        # For each point
        for i in range(len(points)):
            # Skip unlabeled points
            if labels[i] < 0:
                consistency[i] = 0.0
                continue
                
            # Find neighbors within radius
            _, indices, _ = kdtree.search_radius_vector_3d(points[i], radius)
            
            # Skip self
            indices = [idx for idx in indices if idx != i]
            
            # Skip if no neighbors
            if not indices:
                consistency[i] = 1.0
                continue
                
            # Count neighbors with same label
            same_label = sum(1 for idx in indices if labels[idx] == labels[i])
            
            # Compute consistency
            consistency[i] = same_label / len(indices)
        
        return consistency
        
    except ImportError:
        logger.warning("Open3D not available for efficient consistency computation")
        return np.ones(len(points))


def main():
    """Test consensus algorithms on synthetic data."""
    import argparse
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test consensus algorithms")
    parser.add_argument("--strategy", default="ensemble", 
                       choices=["majority", "weighted", "bayesian", "confidence", "ensemble"],
                       help="Consensus strategy to use")
    parser.add_argument("--num_points", type=int, default=1000,
                       help="Number of points to simulate")
    parser.add_argument("--num_labels", type=int, default=5,
                       help="Number of unique labels")
    parser.add_argument("--noise", type=float, default=0.3,
                       help="Noise level (0-1)")
    
    args = parser.parse_args()
    
    # Generate synthetic data
    np.random.seed(42)
    
    # True labels (ground truth)
    true_labels = np.random.randint(0, args.num_labels, args.num_points)
    
    # Generate multiple noisy observations
    num_observations = 5
    observations = []
    
    for _ in range(num_observations):
        # Copy true labels
        noisy_labels = true_labels.copy()
        
        # Add noise
        noise_mask = np.random.random(args.num_points) < args.noise
        noisy_labels[noise_mask] = np.random.randint(0, args.num_labels, np.sum(noise_mask))
        
        # Random confidences
        confidences = np.random.uniform(0.5, 1.0, args.num_points)
        
        # Lower confidence for noisy labels
        confidences[noise_mask] *= 0.7
        
        observations.append((noisy_labels, confidences))
    
    # Create label candidates dictionary
    label_candidates = {}
    for i in range(args.num_points):
        candidates = []
        for obs_idx, (labels, confs) in enumerate(observations):
            # Weight based on observation index (simulating different view reliability)
            weight = 1.0 - 0.1 * obs_idx
            
            candidates.append((int(labels[i]), float(confs[i]), weight))
        
        label_candidates[i] = candidates
    
    # Apply consensus
    consensus_labels, consensus_confidences = apply_consensus(
        label_candidates, args.strategy)
    
    # Compute accuracy
    accuracy = np.mean(consensus_labels == true_labels)
    
    logger.info(f"Consensus strategy: {args.strategy}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot true labels
    plt.subplot(1, 2, 1)
    plt.scatter(range(args.num_points), true_labels, 
               c=true_labels, cmap='tab10', 
               marker='.', label='True Labels')
    plt.title("True Labels")
    plt.xlabel("Point Index")
    plt.ylabel("Label")
    
    # Plot consensus labels
    plt.subplot(1, 2, 2)
    plt.scatter(range(args.num_points), consensus_labels, 
               c=consensus_labels, cmap='tab10', 
               marker='.', label='Consensus Labels')
    plt.title(f"Consensus Labels (Accuracy: {accuracy:.4f})")
    plt.xlabel("Point Index")
    plt.ylabel("Label")
    
    plt.tight_layout()
    plt.savefig(f"consensus_{args.strategy}.png")
    
    logger.info(f"Saved plot to consensus_{args.strategy}.png")


if __name__ == "__main__":
    main()