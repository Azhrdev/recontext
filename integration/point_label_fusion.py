#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point cloud label fusion module.

This module provides methods for fusing semantic labels from multiple sources
into a consistent labeling of 3D point clouds, handling conflicts and uncertainties.

Author: Michael Chen
Date: 2024-02-05
Last modified: 2024-03-12
"""

import numpy as np
import open3d as o3d
import logging
import torch
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from collections import Counter, defaultdict
from dataclasses import dataclass

from recontext.integration.uncertainty import compute_label_uncertainty
from recontext.integration.consensus import create_consensus_strategy
from recontext.semantics.label_manager import LabelManager

logger = logging.getLogger(__name__)

@dataclass
class LabelCandidate:
    """A candidate label for a point."""
    label_id: int
    class_name: str
    confidence: float
    source_id: int
    source_weight: float = 1.0
    
    @property
    def weighted_confidence(self) -> float:
        """Get confidence weighted by source reliability."""
        return self.confidence * self.source_weight


class PointLabelFusion:
    """Fuses semantic labels from multiple sources for 3D points."""
    
    def __init__(self, 
                 consensus_strategy: str = "ensemble",
                 min_confidence: float = 0.5,
                 consider_neighbors: bool = True,
                 neighbor_radius: float = 0.1,
                 smoothing_iterations: int = 1,
                 device: Optional[str] = None):
        """Initialize label fusion.
        
        Args:
            consensus_strategy: Strategy for resolving label conflicts
            min_confidence: Minimum confidence threshold
            consider_neighbors: Whether to consider neighboring points
            neighbor_radius: Radius for neighbor search
            smoothing_iterations: Number of smoothing iterations
            device: Computation device (cpu or cuda)
        """
        self.consensus_strategy = consensus_strategy
        self.min_confidence = min_confidence
        self.consider_neighbors = consider_neighbors
        self.neighbor_radius = neighbor_radius
        self.smoothing_iterations = smoothing_iterations
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Get consensus strategy
        self.consensus = create_consensus_strategy(consensus_strategy)
        
        # Initialize label manager for class info
        self.label_manager = LabelManager()
    
    def fuse_labels(self, 
                   pointcloud: o3d.geometry.PointCloud,
                   label_sources: List[Dict[int, Tuple[int, float]]],
                   source_weights: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fuse labels from multiple sources.
        
        Args:
            pointcloud: The point cloud to label
            label_sources: List of dictionaries mapping point indices to (label_id, confidence) tuples
            source_weights: Optional weights for each source (reliability)
            
        Returns:
            Tuple of (labels, confidences, uncertainties)
        """
        logger.info(f"Fusing labels from {len(label_sources)} sources")
        
        # Get points
        points = np.asarray(pointcloud.points)
        n_points = len(points)
        
        # Initialize output arrays
        labels = np.full(n_points, -1, dtype=np.int32)  # -1 for unlabeled
        confidences = np.zeros(n_points, dtype=np.float32)
        uncertainties = np.ones(n_points, dtype=np.float32)  # 1.0 for max uncertainty
        
        # Set default source weights if not provided
        if source_weights is None:
            source_weights = [1.0] * len(label_sources)
        
        # Collect all label candidates
        candidates_by_point = defaultdict(list)
        
        for source_id, (source_labels, weight) in enumerate(zip(label_sources, source_weights)):
            for point_idx, (label_id, confidence) in source_labels.items():
                if confidence < self.min_confidence:
                    continue
                
                # Get class name
                class_name = self.label_manager.get_class_name(label_id)
                
                # Create candidate
                candidate = LabelCandidate(
                    label_id=label_id,
                    class_name=class_name,
                    confidence=confidence,
                    source_id=source_id,
                    source_weight=weight
                )
                
                candidates_by_point[point_idx].append(candidate)
        
        logger.info(f"Collected candidates for {len(candidates_by_point)} points")
        
        # Build spatial index if considering neighbors
        if self.consider_neighbors:
            kd_tree = o3d.geometry.KDTreeFlann(pointcloud)
        
        # Process each point
        for point_idx in range(n_points):
            # Get candidates for this point
            direct_candidates = candidates_by_point.get(point_idx, [])
            
            # Skip if no direct candidates
            if not direct_candidates:
                continue
            
            # Add candidates from neighboring points if enabled
            if self.consider_neighbors:
                neighbor_candidates = self._get_neighbor_candidates(
                    point_idx, points, kd_tree, candidates_by_point)
                all_candidates = direct_candidates + neighbor_candidates
            else:
                all_candidates = direct_candidates
            
            # Apply consensus strategy
            result_label, result_confidence = self._apply_consensus(all_candidates)
            
            # Store result
            labels[point_idx] = result_label
            confidences[point_idx] = result_confidence
        
        # Apply spatial smoothing if requested
        if self.smoothing_iterations > 0:
            labels, confidences = self._apply_smoothing(
                points, labels, confidences, pointcloud)
        
        # Compute uncertainties
        uncertainties = self._compute_uncertainties(
            points, labels, confidences, candidates_by_point, pointcloud)
        
        # Count labeled points
        labeled_count = np.sum(labels >= 0)
        logger.info(f"Labeled {labeled_count}/{n_points} points ({labeled_count/n_points:.1%})")
        
        return labels, confidences, uncertainties
    
    def _get_neighbor_candidates(self, 
                               point_idx: int,
                               points: np.ndarray,
                               kd_tree: o3d.geometry.KDTreeFlann,
                               candidates_by_point: Dict[int, List[LabelCandidate]]) -> List[LabelCandidate]:
        """Get label candidates from neighboring points.
        
        Args:
            point_idx: Index of the point
            points: Array of point coordinates
            kd_tree: KD-tree for neighbor search
            candidates_by_point: Dictionary of candidates by point index
            
        Returns:
            List of candidates from neighboring points
        """
        # Find neighbors within radius
        _, neighbor_indices, _ = kd_tree.search_radius_vector_3d(
            points[point_idx], self.neighbor_radius)
        
        # Skip self
        neighbor_indices = [idx for idx in neighbor_indices if idx != point_idx]
        
        # Collect candidates from neighbors
        neighbor_candidates = []
        
        for idx in neighbor_indices:
            for candidate in candidates_by_point.get(idx, []):
                # Reduce confidence based on distance
                distance = np.linalg.norm(points[point_idx] - points[idx])
                distance_factor = max(0, 1.0 - distance / self.neighbor_radius)
                
                # Create new candidate with adjusted confidence
                neighbor_candidate = LabelCandidate(
                    label_id=candidate.label_id,
                    class_name=candidate.class_name,
                    confidence=candidate.confidence * distance_factor * 0.8,  # Penalize neighbor contributions
                    source_id=candidate.source_id,
                    source_weight=candidate.source_weight
                )
                
                neighbor_candidates.append(neighbor_candidate)
        
        return neighbor_candidates
    
    def _apply_consensus(self, candidates: List[LabelCandidate]) -> Tuple[int, float]:
        """Apply consensus strategy to resolve label conflicts.
        
        Args:
            candidates: List of label candidates
            
        Returns:
            Tuple of (selected_label_id, confidence)
        """
        if not candidates:
            return -1, 0.0
        
        # Group candidates by label
        candidates_by_label = defaultdict(list)
        for candidate in candidates:
            candidates_by_label[candidate.label_id].append(candidate)
        
        # Prepare inputs for consensus strategy
        label_confs = []
        weights = []
        
        for label_id, label_candidates in candidates_by_label.items():
            # Compute average confidence for this label
            avg_conf = sum(c.weighted_confidence for c in label_candidates) / len(label_candidates)
            
            # Compute weight based on number of sources agreeing on this label
            source_ids = set(c.source_id for c in label_candidates)
            weight = len(source_ids)
            
            label_confs.append((label_id, avg_conf))
            weights.append(weight)
        
        # Apply consensus strategy
        result_label, result_confidence = self.consensus.resolve(label_confs, weights)
        
        return result_label, result_confidence
    
    def _apply_smoothing(self, 
                        points: np.ndarray,
                        labels: np.ndarray,
                        confidences: np.ndarray,
                        pointcloud: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial smoothing to labels.
        
        Args:
            points: Array of point coordinates
            labels: Array of point labels
            confidences: Array of label confidences
            pointcloud: Original point cloud
            
        Returns:
            Tuple of (smoothed_labels, smoothed_confidences)
        """
        logger.info(f"Applying {self.smoothing_iterations} smoothing iterations")
        
        # Create KD-tree
        kd_tree = o3d.geometry.KDTreeFlann(pointcloud)
        
        # Initialize smoothed arrays
        smoothed_labels = labels.copy()
        smoothed_confidences = confidences.copy()
        
        # Apply multiple iterations
        for iteration in range(self.smoothing_iterations):
            next_labels = smoothed_labels.copy()
            next_confidences = smoothed_confidences.copy()
            
            # Process each point
            for i in range(len(points)):
                # Skip unlabeled points
                if smoothed_labels[i] < 0:
                    continue
                
                # Find neighbors
                _, neighbor_indices, _ = kd_tree.search_radius_vector_3d(
                    points[i], self.neighbor_radius)
                
                # Skip if no neighbors
                if len(neighbor_indices) <= 1:
                    continue
                
                # Get labeled neighbors
                labeled_neighbors = []
                for idx in neighbor_indices:
                    if idx != i and smoothed_labels[idx] >= 0:
                        labeled_neighbors.append(idx)
                
                if not labeled_neighbors:
                    continue
                
                # Count neighbor labels
                label_counts = Counter([smoothed_labels[idx] for idx in labeled_neighbors])
                
                # Get majority label
                majority_label, count = label_counts.most_common(1)[0]
                
                # Update label if majority is different and has enough support
                if majority_label != smoothed_labels[i] and count >= len(labeled_neighbors) / 2:
                    next_labels[i] = majority_label
                    
                    # Adjust confidence
                    majority_conf = sum(smoothed_confidences[idx] 
                                     for idx in labeled_neighbors 
                                     if smoothed_labels[idx] == majority_label)
                    majority_conf /= count
                    
                    next_confidences[i] = (smoothed_confidences[i] + majority_conf) / 2
            
            # Update for next iteration
            smoothed_labels = next_labels
            smoothed_confidences = next_confidences
            
            # Count changes
            changes = np.sum(smoothed_labels != labels)
            logger.info(f"Iteration {iteration+1}: {changes} label changes")
        
        return smoothed_labels, smoothed_confidences
    
    def _compute_uncertainties(self, 
                              points: np.ndarray,
                              labels: np.ndarray,
                              confidences: np.ndarray,
                              candidates_by_point: Dict[int, List[LabelCandidate]],
                              pointcloud: o3d.geometry.PointCloud) -> np.ndarray:
        """Compute uncertainty values for each point.
        
        Args:
            points: Array of point coordinates
            labels: Array of point labels
            confidences: Array of label confidences
            candidates_by_point: Dictionary of candidates by point index
            pointcloud: Original point cloud
            
        Returns:
            Array of uncertainty values
        """
        logger.info("Computing label uncertainties")
        
        # Initialize with inverse of confidence
        uncertainties = 1.0 - np.clip(confidences, 0, 1)
        
        # Create dictionary mapping point indices to lists of (label_id, confidence, weight) tuples
        label_candidates = {}
        for point_idx, candidates in candidates_by_point.items():
            # Convert to format expected by compute_label_uncertainty
            label_candidates[point_idx] = [
                (c.label_id, c.confidence, c.source_weight) for c in candidates
            ]
        
        # Compute label uncertainty based on label distribution
        point_view_counts = np.zeros(len(points), dtype=np.int32)
        
        for point_idx, candidates in label_candidates.items():
            point_view_counts[point_idx] = len(set(c[0] for c in candidates))
        
        # Use uncertainty module
        computed_uncertainties = compute_label_uncertainty(label_candidates, point_view_counts)
        
        # Combine uncertainty metrics (use computed where available)
        mask = computed_uncertainties > 0
        uncertainties[mask] = computed_uncertainties[mask]
        
        # Add spatial consistency factor
        if self.consider_neighbors:
            # Create KD-tree
            kd_tree = o3d.geometry.KDTreeFlann(pointcloud)
            
            # Process each labeled point
            for i in range(len(points)):
                if labels[i] < 0:
                    continue
                
                # Find neighbors
                _, neighbor_indices, _ = kd_tree.search_radius_vector_3d(
                    points[i], self.neighbor_radius)
                
                # Skip self and unlabeled neighbors
                labeled_neighbors = [idx for idx in neighbor_indices 
                                  if idx != i and labels[idx] >= 0]
                
                if not labeled_neighbors:
                    continue
                
                # Count neighbors with same label
                same_label = sum(1 for idx in labeled_neighbors if labels[idx] == labels[i])
                
                # Compute consistency factor
                consistency = same_label / len(labeled_neighbors)
                
                # Adjust uncertainty based on consistency
                uncertainties[i] *= (1.0 - consistency * 0.5)  # Reduce by up to 50% for perfect consistency
        
        return uncertainties
    
    def create_colored_pointcloud(self, 
                                 pointcloud: o3d.geometry.PointCloud,
                                 labels: np.ndarray,
                                 confidences: np.ndarray,
                                 uncertainties: np.ndarray) -> o3d.geometry.PointCloud:
        """Create a colored point cloud based on labels.
        
        Args:
            pointcloud: Original point cloud
            labels: Array of point labels
            confidences: Array of label confidences
            uncertainties: Array of label uncertainties
            
        Returns:
            Colored point cloud
        """
        # Create a copy of the point cloud
        colored_cloud = o3d.geometry.PointCloud(pointcloud)
        
        # Get number of points
        n_points = len(pointcloud.points)
        
        # Initialize colors
        colors = np.zeros((n_points, 3))
        
        # Get color for each label
        for i in range(n_points):
            label = labels[i]
            
            if label >= 0:
                # Get color from label manager
                color = self.label_manager.get_color(label)
                
                # Adjust saturation based on confidence
                conf = confidences[i]
                saturation = max(0.3, conf)
                
                # Adjust brightness based on uncertainty
                uncertainty = uncertainties[i]
                brightness = max(0.5, 1.0 - uncertainty * 0.5)
                
                # Normalize to [0, 1]
                color_norm = np.array(color) / 255.0 * saturation * brightness
                
                colors[i] = color_norm
            else:
                # Gray for unlabeled points
                colors[i] = [0.7, 0.7, 0.7]
        
        # Set point colors
        colored_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        return colored_cloud