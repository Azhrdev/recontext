#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Label projection module for transferring 2D semantic labels to 3D points.

This module handles the projection of 2D semantic segmentation masks onto 3D point clouds
and meshes, establishing a link between 2D semantic understanding and 3D geometry.

Author: Alex Johnson
Date: 2024-02-12
"""

import numpy as np
import open3d as o3d
import torch
import logging
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
from tqdm import tqdm

from recontext.semantics.instance_segmentation import InstanceData
from recontext.integration.uncertainty import compute_label_uncertainty
from recontext.utils.transforms import world_to_camera, camera_to_pixel

logger = logging.getLogger(__name__)

@dataclass
class ProjectedLabel:
    """Data structure for projected semantic labels."""
    point_idx: int  # Index of 3D point
    instance_id: int  # ID of the instance
    class_id: int  # Class ID
    class_name: str  # Class name
    score: float  # Confidence score
    color: Tuple[int, int, int]  # RGB color
    view_count: int = 0  # Number of views contributing to this label
    weights: List[float] = None  # Weights from different views


class LabelProjector:
    """Projects 2D semantic segmentation labels onto 3D geometry."""
    
    def __init__(self, 
                 visibility_threshold: float = 0.1,
                 min_views_per_point: int = 2,
                 max_projection_distance: float = 0.05,
                 use_weighted_consensus: bool = True):
        """Initialize label projector.
        
        Args:
            visibility_threshold: Threshold for point visibility check
            min_views_per_point: Minimum number of views required for a point to be labeled
            max_projection_distance: Maximum projection distance for mesh vertices
            use_weighted_consensus: Whether to use weighted consensus for conflicting labels
        """
        self.visibility_threshold = visibility_threshold
        self.min_views_per_point = min_views_per_point
        self.max_projection_distance = max_projection_distance
        self.use_weighted_consensus = use_weighted_consensus
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def project_labels(self, 
                       pointcloud: o3d.geometry.PointCloud,
                       cameras: List[Dict],
                       segmentations: List[List[InstanceData]]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Project 2D instance segmentations onto 3D point cloud.
        
        Args:
            pointcloud: 3D point cloud
            cameras: List of camera parameters
            segmentations: List of instance segmentations for each camera view
            
        Returns:
            Tuple containing:
                - Array of point labels (class IDs)
                - Array of point colors
                - List of label metadata
        """
        if len(cameras) != len(segmentations):
            raise ValueError(f"Number of cameras ({len(cameras)}) must match number of segmentations ({len(segmentations)})")
        
        # Extract points from point cloud
        points = np.asarray(pointcloud.points)
        n_points = len(points)
        
        logger.info(f"Projecting labels from {len(cameras)} views to {n_points} points")
        
        # Initialize label containers
        point_labels = -np.ones(n_points, dtype=int)  # Class IDs (-1 for unlabeled)
        point_colors = np.zeros((n_points, 3), dtype=np.uint8)  # RGB colors
        point_scores = np.zeros(n_points, dtype=float)  # Confidence scores
        point_view_counts = np.zeros(n_points, dtype=int)  # Number of views per point
        point_instance_ids = -np.ones(n_points, dtype=int)  # Instance IDs (-1 for unlabeled)
        
        # Label candidates for each point
        label_candidates = defaultdict(list)  # point_idx -> list of ProjectedLabel
        
        # Process each camera view
        for cam_idx, (camera, instances) in enumerate(tqdm(zip(cameras, segmentations), 
                                                        desc="Projecting labels",
                                                        total=len(cameras))):
            if not instances:
                logger.debug(f"No instances in view {cam_idx}, skipping")
                continue
                
            # Get camera parameters
            extrinsic = camera.get('extrinsic', np.eye(4))  # 4x4 camera pose
            intrinsic = camera.get('intrinsic', np.eye(3))  # 3x3 camera intrinsic
            image_width = camera.get('width', 1920)
            image_height = camera.get('height', 1080)
            
            # Convert to torch for faster processing
            points_torch = torch.tensor(points, device=self.device, dtype=torch.float32)
            extrinsic_torch = torch.tensor(extrinsic, device=self.device, dtype=torch.float32)
            intrinsic_torch = torch.tensor(intrinsic, device=self.device, dtype=torch.float32)
            
            # Project 3D points to camera space
            cam_points = world_to_camera(points_torch, extrinsic_torch)
            
            # Filter points in front of camera
            depth_mask = cam_points[:, 2] > 0
            if torch.sum(depth_mask) == 0:
                logger.debug(f"No points in front of camera {cam_idx}, skipping")
                continue
            
            # Project to pixel coordinates
            pixels = camera_to_pixel(cam_points, intrinsic_torch)
            
            # Filter points within image bounds
            valid_mask = (
                (pixels[:, 0] >= 0) & (pixels[:, 0] < image_width) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < image_height) &
                depth_mask
            )
            
            valid_indices = torch.where(valid_mask)[0].cpu().numpy()
            valid_pixels = pixels[valid_mask].cpu().numpy().astype(int)
            
            if len(valid_indices) == 0:
                logger.debug(f"No valid projections for camera {cam_idx}, skipping")
                continue
                
            logger.debug(f"Camera {cam_idx}: {len(valid_indices)} points project into view")
            
            # Check each instance mask for point containment
            for instance_id, instance in enumerate(instances):
                mask = instance.mask  # Binary mask
                
                # Check which valid points fall within the mask
                mask_values = mask[valid_pixels[:, 1], valid_pixels[:, 0]]
                instance_point_indices = valid_indices[mask_values > 0]
                
                if len(instance_point_indices) == 0:
                    continue
                    
                # Distance from camera (as weight for voting)
                distances = cam_points[instance_point_indices, 2].cpu().numpy()
                weights = 1.0 / (distances + 1e-6)  # Inverse distance
                weights = weights / np.max(weights)  # Normalize weights
                
                # Create projected labels for points in this instance
                for i, point_idx in enumerate(instance_point_indices):
                    label = ProjectedLabel(
                        point_idx=int(point_idx),
                        instance_id=instance_id,
                        class_id=instance.class_id,
                        class_name=instance.class_name,
                        score=instance.score,
                        color=instance.color,
                        view_count=1,
                        weights=[float(weights[i])]
                    )
                    
                    label_candidates[point_idx].append(label)
                    point_view_counts[point_idx] += 1
        
        logger.info(f"Found {sum(len(candidates) for candidates in label_candidates.values())} label candidates")
        logger.info(f"{len(label_candidates)} points have at least one label")
        
        # Resolve conflicts using consensus voting
        for point_idx, candidates in label_candidates.items():
            if point_view_counts[point_idx] < self.min_views_per_point:
                continue
                
            if len(candidates) == 1:
                # Only one candidate, use it directly
                label = candidates[0]
                point_labels[point_idx] = label.class_id
                point_colors[point_idx] = label.color
                point_scores[point_idx] = label.score
                point_instance_ids[point_idx] = label.instance_id
            else:
                # Multiple candidates, use voting
                if self.use_weighted_consensus:
                    # Group by class and compute weighted votes
                    class_votes = defaultdict(float)
                    for label in candidates:
                        class_votes[label.class_id] += sum(label.weights)
                else:
                    # Simple majority voting
                    class_votes = Counter([label.class_id for label in candidates])
                
                # Get winning class
                winner_class = max(class_votes, key=class_votes.get)
                
                # Find best instance of winning class
                winner_instances = [label for label in candidates if label.class_id == winner_class]
                winner = max(winner_instances, key=lambda x: x.score)
                
                point_labels[point_idx] = winner.class_id
                point_colors[point_idx] = winner.color
                point_scores[point_idx] = winner.score
                point_instance_ids[point_idx] = winner.instance_id
        
        # Compute uncertainty
        uncertainty = compute_label_uncertainty(label_candidates, point_view_counts)
        
        # Create metadata for each unique class
        class_metadata = {}
        for candidates in label_candidates.values():
            for label in candidates:
                if label.class_id not in class_metadata:
                    class_metadata[label.class_id] = {
                        'class_id': label.class_id,
                        'class_name': label.class_name,
                        'color': label.color,
                        'count': 0
                    }
        
        # Count points per class
        for class_id in point_labels:
            if class_id >= 0 and class_id in class_metadata:
                class_metadata[class_id]['count'] += 1
        
        # Convert to list and sort by class_id
        metadata = [class_metadata[class_id] for class_id in sorted(class_metadata.keys())]
        
        # Calculate statistics
        labeled_points = np.sum(point_labels >= 0)
        label_ratio = labeled_points / n_points if n_points > 0 else 0
        
        logger.info(f"Labeled {labeled_points}/{n_points} points ({label_ratio:.2%})")
        logger.info(f"Found {len(metadata)} unique classes")
        
        return point_labels, point_colors, metadata
    
    def project_to_mesh(self, 
                        mesh: o3d.geometry.TriangleMesh,
                        pointcloud: o3d.geometry.PointCloud,
                        point_labels: np.ndarray,
                        point_colors: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Project point cloud labels onto mesh.
        
        Args:
            mesh: Target mesh
            pointcloud: Labeled point cloud
            point_labels: Point labels (class IDs)
            point_colors: Point colors
            
        Returns:
            Labeled mesh
        """
        logger.info("Projecting labels from point cloud to mesh")
        
        # Get mesh vertices and create a KD-tree from the point cloud
        mesh_vertices = np.asarray(mesh.vertices)
        points = np.asarray(pointcloud.points)
        
        # Create labeled mesh copy
        labeled_mesh = o3d.geometry.TriangleMesh(mesh)
        
        # Create KD-tree
        pcd_tree = o3d.geometry.KDTreeFlann(pointcloud)
        
        # Initialize vertex colors and labels
        vertex_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float64)
        vertex_labels = -np.ones(len(mesh_vertices), dtype=int)
        
        # Search for nearest neighbors for each vertex
        for i, vertex in enumerate(tqdm(mesh_vertices, desc="Projecting to mesh")):
            # Find k nearest neighbors
            k = 5  # Number of neighbors to consider
            _, indices, distances = pcd_tree.search_knn_vector_3d(vertex, k)
            
            # Filter neighbors by distance
            valid_indices = [idx for idx, dist in zip(indices, distances) 
                           if dist <= self.max_projection_distance ** 2]
            
            if not valid_indices:
                continue
                
            # Get labels and colors of valid neighbors
            neighbor_labels = [point_labels[idx] for idx in valid_indices]
            neighbor_colors = [point_colors[idx] for idx in valid_indices]
            
            # Filter out unlabeled points
            labeled_indices = [i for i, label in enumerate(neighbor_labels) if label >= 0]
            
            if not labeled_indices:
                continue
                
            # Use majority voting for label
            label_counts = Counter([neighbor_labels[i] for i in labeled_indices])
            majority_label = label_counts.most_common(1)[0][0]
            
            # Assign label and color to vertex
            vertex_labels[i] = majority_label
            
            # Average the colors of points with the majority label
            majority_indices = [labeled_indices[j] for j, idx in enumerate(labeled_indices) 
                              if neighbor_labels[idx] == majority_label]
            
            if majority_indices:
                majority_colors = [neighbor_colors[idx] for idx in majority_indices]
                vertex_colors[i] = np.mean(majority_colors, axis=0) / 255.0  # Normalize to [0,1]
        
        # Assign colors to mesh
        labeled_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        # Calculate statistics
        labeled_vertices = np.sum(vertex_labels >= 0)
        label_ratio = labeled_vertices / len(mesh_vertices) if len(mesh_vertices) > 0 else 0
        
        logger.info(f"Labeled {labeled_vertices}/{len(mesh_vertices)} vertices ({label_ratio:.2%})")
        
        return labeled_mesh
    
    @staticmethod
    def visualize_labels(pointcloud: o3d.geometry.PointCloud, 
                        point_labels: np.ndarray,
                        metadata: List[Dict]) -> o3d.geometry.PointCloud:
        """Create a colored point cloud for label visualization.
        
        Args:
            pointcloud: Original point cloud
            point_labels: Point labels (class IDs)
            metadata: Label metadata containing colors
            
        Returns:
            Colored point cloud
        """
        # Create a copy of the point cloud
        labeled_pcd = o3d.geometry.PointCloud(pointcloud)
        
        # Get points
        points = np.asarray(pointcloud.points)
        n_points = len(points)
        
        # Create color map from metadata
        color_map = {meta['class_id']: np.array(meta['color']) / 255.0 for meta in metadata}
        
        # Default color for unlabeled points (gray)
        default_color = np.array([0.7, 0.7, 0.7])
        
        # Create color array
        colors = np.zeros((n_points, 3))
        
        # Assign colors based on labels
        for i in range(n_points):
            label = point_labels[i]
            if label >= 0 and label in color_map:
                colors[i] = color_map[label]
            else:
                colors[i] = default_color
        
        # Assign colors to point cloud
        labeled_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return labeled_pcd


def main():
    """Main function for standalone usage."""
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description="Project 2D labels to 3D")
    parser.add_argument("--pointcloud", required=True, help="Input point cloud file (.ply)")
    parser.add_argument("--cameras", required=True, help="Camera parameters file (.pkl)")
    parser.add_argument("--segmentations", required=True, help="Segmentation results file (.pkl)")
    parser.add_argument("--output", required=True, help="Output labeled point cloud (.ply)")
    parser.add_argument("--min-views", type=int, default=2, help="Minimum number of views per point")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load point cloud
    logger.info(f"Loading point cloud from {args.pointcloud}")
    pointcloud = o3d.io.read_point_cloud(args.pointcloud)
    
    # Load camera parameters
    logger.info(f"Loading camera parameters from {args.cameras}")
    with open(args.cameras, 'rb') as f:
        cameras = pickle.load(f)
    
    # Load segmentation results
    logger.info(f"Loading segmentation results from {args.segmentations}")
    with open(args.segmentations, 'rb') as f:
        segmentations = pickle.load(f)
    
    # Create label projector
    projector = LabelProjector(min_views_per_point=args.min_views)
    
    # Project labels
    point_labels, point_colors, metadata = projector.project_labels(
        pointcloud, cameras, segmentations)
    
    # Visualize labels
    labeled_pcd = projector.visualize_labels(pointcloud, point_labels, metadata)
    
    # Save labeled point cloud
    logger.info(f"Saving labeled point cloud to {args.output}")
    o3d.io.write_point_cloud(args.output, labeled_pcd)
    
    # Print statistics
    labeled_points = np.sum(point_labels >= 0)
    total_points = len(point_labels)
    logger.info(f"Labeled {labeled_points}/{total_points} points " +
               f"({labeled_points/total_points:.2%})")
    
    class_counts = Counter(point_labels[point_labels >= 0])
    logger.info(f"Found {len(class_counts)} unique classes")
    
    # Print top 5 classes by count
    top_classes = class_counts.most_common(5)
    for class_id, count in top_classes:
        class_name = next((m['class_name'] for m in metadata if m['class_id'] == class_id), "unknown")
        logger.info(f"  Class {class_id} ({class_name}): {count} points " +
                   f"({count/labeled_points:.2%} of labeled)")


if __name__ == "__main__":
    main()