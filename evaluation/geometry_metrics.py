#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geometric evaluation metrics for 3D reconstruction.

This module implements metrics for evaluating the quality of 3D reconstructions
including point clouds, meshes, and camera poses.

Author: Sarah Li
Date: 2024-01-30
Last modified: 2024-03-10
"""

import numpy as np
import open3d as o3d
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy.spatial import cKDTree
import math
import time

logger = logging.getLogger(__name__)

def compute_pointcloud_metrics(prediction: o3d.geometry.PointCloud, 
                             ground_truth: o3d.geometry.PointCloud,
                             downsample: bool = True,
                             voxel_size: float = 0.01,
                             max_distance: float = 0.1) -> Dict[str, float]:
    """Compute metrics between predicted and ground truth point clouds.
    
    Args:
        prediction: Predicted point cloud
        ground_truth: Ground truth point cloud
        downsample: Whether to downsample point clouds before evaluation
        voxel_size: Voxel size for downsampling
        max_distance: Maximum distance threshold for precision/recall
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing point cloud metrics")
    start_time = time.time()
    
    # Ensure we have non-empty point clouds
    if not prediction.has_points() or not ground_truth.has_points():
        logger.error("Empty point cloud(s) provided for evaluation")
        return {
            'chamfer_distance': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'hausdorff_distance': float('inf'),
            'surface_accuracy': 0.0,
            'surface_completeness': 0.0,
            'runtime': 0.0
        }
    
    # Downsample if requested
    if downsample:
        pred_down = prediction.voxel_down_sample(voxel_size)
        gt_down = ground_truth.voxel_down_sample(voxel_size)
    else:
        pred_down = prediction
        gt_down = ground_truth
    
    # Extract points
    pred_points = np.asarray(pred_down.points)
    gt_points = np.asarray(gt_down.points)
    
    # Build KD-trees
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    
    # Compute distances
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    
    # Compute Chamfer distance
    chamfer = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
    
    # Compute precision (percentage of predicted points within threshold of ground truth)
    precision = np.mean(dist_pred_to_gt < max_distance)
    
    # Compute recall (percentage of ground truth points within threshold of prediction)
    recall = np.mean(dist_gt_to_pred < max_distance)
    
    # Compute F1 score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Compute Hausdorff distance (maximum distance between the two point clouds)
    hausdorff = max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))
    
    # Surface accuracy (mean distance from prediction to ground truth)
    surface_accuracy = np.mean(dist_pred_to_gt)
    
    # Surface completeness (mean distance from ground truth to prediction)
    surface_completeness = np.mean(dist_gt_to_pred)
    
    runtime = time.time() - start_time
    
    metrics = {
        'chamfer_distance': float(chamfer),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'hausdorff_distance': float(hausdorff),
        'surface_accuracy': float(surface_accuracy),
        'surface_completeness': float(surface_completeness),
        'runtime': float(runtime)
    }
    
    logger.info(f"Point cloud metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_mesh_metrics(prediction: o3d.geometry.TriangleMesh, 
                        ground_truth: o3d.geometry.TriangleMesh,
                        samples: int = 100000,
                        max_distance: float = 0.1) -> Dict[str, float]:
    """Compute metrics between predicted and ground truth meshes.
    
    Args:
        prediction: Predicted mesh
        ground_truth: Ground truth mesh
        samples: Number of points to sample for evaluation
        max_distance: Maximum distance threshold for precision/recall
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing mesh metrics")
    start_time = time.time()
    
    # Ensure we have non-empty meshes
    if not prediction.has_triangles() or not ground_truth.has_triangles():
        logger.error("Empty mesh(es) provided for evaluation")
        return {
            'chamfer_distance': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'hausdorff_distance': float('inf'),
            'surface_accuracy': 0.0,
            'surface_completeness': 0.0,
            'runtime': 0.0
        }
    
    try:
        # Sample points from meshes
        pred_pcd = prediction.sample_points_uniformly(samples)
        gt_pcd = ground_truth.sample_points_uniformly(samples)
        
        # Extract points
        pred_points = np.asarray(pred_pcd.points)
        gt_points = np.asarray(gt_pcd.points)
        
        # Build KD-trees
        pred_tree = cKDTree(pred_points)
        gt_tree = cKDTree(gt_points)
        
        # Compute distances
        dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
        dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
        
        # Compute Chamfer distance
        chamfer = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)
        
        # Compute precision (percentage of predicted points within threshold of ground truth)
        precision = np.mean(dist_pred_to_gt < max_distance)
        
        # Compute recall (percentage of ground truth points within threshold of prediction)
        recall = np.mean(dist_gt_to_pred < max_distance)
        
        # Compute F1 score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute Hausdorff distance (maximum distance between the two point clouds)
        hausdorff = max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))
        
        # Surface accuracy (mean distance from prediction to ground truth)
        surface_accuracy = np.mean(dist_pred_to_gt)
        
        # Surface completeness (mean distance from ground truth to prediction)
        surface_completeness = np.mean(dist_gt_to_pred)
        
        runtime = time.time() - start_time
        
        metrics = {
            'chamfer_distance': float(chamfer),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'hausdorff_distance': float(hausdorff),
            'surface_accuracy': float(surface_accuracy),
            'surface_completeness': float(surface_completeness),
            'runtime': float(runtime)
        }
        
        logger.info(f"Mesh metrics computed in {runtime:.2f} seconds")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing mesh metrics: {e}")
        return {
            'chamfer_distance': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'hausdorff_distance': float('inf'),
            'surface_accuracy': 0.0,
            'surface_completeness': 0.0,
            'runtime': 0.0
        }


def compute_camera_pose_metrics(pred_cameras: Dict[int, Any], 
                              gt_cameras: Dict[int, Any]) -> Dict[str, float]:
    """Compute metrics between predicted and ground truth camera poses.
    
    Args:
        pred_cameras: Dictionary of predicted cameras {image_id: camera}
        gt_cameras: Dictionary of ground truth cameras {image_id: camera}
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing camera pose metrics")
    start_time = time.time()
    
    # Check if we have matching cameras
    common_ids = set(pred_cameras.keys()) & set(gt_cameras.keys())
    if not common_ids:
        logger.error("No common cameras found for evaluation")
        return {
            'rotation_error_mean': float('inf'),
            'rotation_error_median': float('inf'),
            'translation_error_mean': float('inf'),
            'translation_error_median': float('inf'),
            'pose_error_mean': float('inf'),
            'pose_error_median': float('inf'),
            'runtime': 0.0
        }
    
    # Calculate errors
    rotation_errors = []
    translation_errors = []
    pose_errors = []
    
    for image_id in common_ids:
        pred_camera = pred_cameras[image_id]
        gt_camera = gt_cameras[image_id]
        
        # Extract rotation and translation
        pred_R = pred_camera.extrinsics.R
        pred_t = pred_camera.extrinsics.t
        gt_R = gt_camera.extrinsics.R
        gt_t = gt_camera.extrinsics.t
        
        # Compute rotation error (in degrees)
        R_diff = np.dot(pred_R, gt_R.T)
        trace = np.trace(R_diff)
        trace = min(3.0, max(-1.0, trace))  # Clamp to handle numerical errors
        angle_error = np.degrees(np.arccos((trace - 1) / 2))
        rotation_errors.append(angle_error)
        
        # Compute translation error (Euclidean distance)
        trans_error = np.linalg.norm(pred_t - gt_t)
        translation_errors.append(trans_error)
        
        # Combined pose error
        pose_error = angle_error + trans_error  # Simple weighted sum
        pose_errors.append(pose_error)
    
    # Compute statistics
    rotation_error_mean = np.mean(rotation_errors)
    rotation_error_median = np.median(rotation_errors)
    translation_error_mean = np.mean(translation_errors)
    translation_error_median = np.median(translation_errors)
    pose_error_mean = np.mean(pose_errors)
    pose_error_median = np.median(pose_errors)
    
    runtime = time.time() - start_time
    
    metrics = {
        'rotation_error_mean': float(rotation_error_mean),
        'rotation_error_median': float(rotation_error_median),
        'translation_error_mean': float(translation_error_mean),
        'translation_error_median': float(translation_error_median),
        'pose_error_mean': float(pose_error_mean),
        'pose_error_median': float(pose_error_median),
        'runtime': float(runtime)
    }
    
    logger.info(f"Camera pose metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_depth_map_metrics(pred_depth: np.ndarray,
                            gt_depth: np.ndarray,
                            mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute metrics between predicted and ground truth depth maps.
    
    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        mask: Optional mask of valid pixels
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing depth map metrics")
    start_time = time.time()
    
    # Apply mask if provided
    if mask is not None:
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
    else:
        # Create mask for valid depth values
        mask = (gt_depth > 0) & (pred_depth > 0)
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
    
    # Check if we have valid depth values
    if len(pred_depth) == 0 or len(gt_depth) == 0:
        logger.error("No valid depth values found for evaluation")
        return {
            'rmse': float('inf'),
            'abs_rel': float('inf'),
            'sq_rel': float('inf'),
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0,
            'runtime': 0.0
        }
    
    # Compute metrics
    abs_diff = np.abs(pred_depth - gt_depth)
    sq_diff = (pred_depth - gt_depth) ** 2
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean(sq_diff))
    
    # Absolute Relative Error
    abs_rel = np.mean(abs_diff / gt_depth)
    
    # Squared Relative Error
    sq_rel = np.mean(sq_diff / gt_depth)
    
    # Thresholded Accuracy
    max_ratio = np.maximum(pred_depth / gt_depth, gt_depth / pred_depth)
    a1 = np.mean(max_ratio < 1.25)
    a2 = np.mean(max_ratio < 1.25**2)
    a3 = np.mean(max_ratio < 1.25**3)
    
    runtime = time.time() - start_time
    
    metrics = {
        'rmse': float(rmse),
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'a1': float(a1),
        'a2': float(a2),
        'a3': float(a3),
        'runtime': float(runtime)
    }
    
    logger.info(f"Depth map metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_normal_consistency(pred_mesh: o3d.geometry.TriangleMesh,
                             gt_mesh: o3d.geometry.TriangleMesh,
                             samples: int = 100000) -> float:
    """Compute normal consistency between predicted and ground truth meshes.
    
    Args:
        pred_mesh: Predicted mesh
        gt_mesh: Ground truth mesh
        samples: Number of points to sample for evaluation
        
    Returns:
        Normal consistency score (higher is better)
    """
    try:
        # Ensure normals are computed
        pred_mesh.compute_vertex_normals()
        gt_mesh.compute_vertex_normals()
        
        # Sample points with normals
        pred_pcd = pred_mesh.sample_points_poisson_disk(samples, use_triangle_normal=True)
        gt_pcd = gt_mesh.sample_points_poisson_disk(samples, use_triangle_normal=True)
        
        # Extract points and normals
        pred_points = np.asarray(pred_pcd.points)
        pred_normals = np.asarray(pred_pcd.normals)
        gt_points = np.asarray(gt_pcd.points)
        gt_normals = np.asarray(gt_pcd.normals)
        
        # Build KD-tree for ground truth
        gt_tree = cKDTree(gt_points)
        
        # Find closest ground truth points
        _, indices = gt_tree.query(pred_points, k=1)
        
        # Get corresponding normals
        corresponding_gt_normals = gt_normals[indices]
        
        # Compute consistency (dot product of normalized normals)
        # Value of 1 means perfectly aligned, -1 means opposite directions
        dot_products = np.sum(pred_normals * corresponding_gt_normals, axis=1)
        
        # Take absolute value as normals could be flipped
        abs_dot_products = np.abs(dot_products)
        
        # Compute consistency (mean of absolute dot products)
        consistency = np.mean(abs_dot_products)
        
        return float(consistency)
        
    except Exception as e:
        logger.error(f"Error computing normal consistency: {e}")
        return 0.0


def compute_all_geometry_metrics(prediction: Dict[str, Any],
                               ground_truth: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute all geometric metrics for a reconstruction.
    
    Args:
        prediction: Dictionary of prediction results
        ground_truth: Dictionary of ground truth data
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Point cloud metrics
    if 'pointcloud' in prediction and 'pointcloud' in ground_truth:
        metrics['pointcloud'] = compute_pointcloud_metrics(
            prediction['pointcloud'],
            ground_truth['pointcloud']
        )
    
    # Mesh metrics
    if 'mesh' in prediction and 'mesh' in ground_truth:
        metrics['mesh'] = compute_mesh_metrics(
            prediction['mesh'],
            ground_truth['mesh']
        )
        
        # Add normal consistency
        metrics['mesh']['normal_consistency'] = compute_normal_consistency(
            prediction['mesh'],
            ground_truth['mesh']
        )
    
    # Camera pose metrics
    if 'cameras' in prediction and 'cameras' in ground_truth:
        metrics['cameras'] = compute_camera_pose_metrics(
            prediction['cameras'],
            ground_truth['cameras']
        )
    
    # Depth map metrics
    if 'depth_maps' in prediction and 'depth_maps' in ground_truth:
        # Compute metrics for each depth map and average
        depth_metrics_list = []
        for pred_depth, gt_depth in zip(prediction['depth_maps'], ground_truth['depth_maps']):
            if pred_depth is not None and gt_depth is not None:
                metrics_single = compute_depth_map_metrics(pred_depth, gt_depth)
                depth_metrics_list.append(metrics_single)
        
        if depth_metrics_list:
            # Compute average metrics
            depth_metrics = {}
            for key in depth_metrics_list[0].keys():
                depth_metrics[key] = np.mean([m[key] for m in depth_metrics_list])
            
            metrics['depth'] = depth_metrics
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compute geometric evaluation metrics")
    parser.add_argument("--pred_pcd", help="Path to predicted point cloud")
    parser.add_argument("--gt_pcd", help="Path to ground truth point cloud")
    parser.add_argument("--pred_mesh", help="Path to predicted mesh")
    parser.add_argument("--gt_mesh", help="Path to ground truth mesh")
    parser.add_argument("--output", help="Path to save metrics")
    
    args = parser.parse_args()
    
    # Compute point cloud metrics if provided
    if args.pred_pcd and args.gt_pcd:
        logger.info(f"Loading point clouds: {args.pred_pcd}, {args.gt_pcd}")
        pred_pcd = o3d.io.read_point_cloud(args.pred_pcd)
        gt_pcd = o3d.io.read_point_cloud(args.gt_pcd)
        
        metrics = compute_pointcloud_metrics(pred_pcd, gt_pcd)
        logger.info("Point cloud metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
            
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump({'pointcloud': metrics}, f, indent=2)
    
    # Compute mesh metrics if provided
    if args.pred_mesh and args.gt_mesh:
        logger.info(f"Loading meshes: {args.pred_mesh}, {args.gt_mesh}")
        pred_mesh = o3d.io.read_triangle_mesh(args.pred_mesh)
        gt_mesh = o3d.io.read_triangle_mesh(args.gt_mesh)
        
        metrics = compute_mesh_metrics(pred_mesh, gt_mesh)
        normal_consistency = compute_normal_consistency(pred_mesh, gt_mesh)
        metrics['normal_consistency'] = normal_consistency
        
        logger.info("Mesh metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
            
        if args.output:
            import json
            output_path = args.output if not (args.pred_pcd and args.gt_pcd) else args.output.replace('.json', '_mesh.json')
            with open(output_path, 'w') as f:
                json.dump({'mesh': metrics}, f, indent=2)