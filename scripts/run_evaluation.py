#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for RECONTEXT.

This script runs evaluation on 3D reconstruction and semantic understanding results,
computing various metrics for geometric and semantic quality.

Author: Sarah Li
Date: 2024-02-10
Last modified: 2024-03-15
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recontext.evaluation.geometry_metrics import (
    compute_pointcloud_metrics,
    compute_mesh_metrics,
    compute_camera_pose_metrics,
    compute_normal_consistency
)
from recontext.evaluation.semantic_metrics import (
    compute_semantic_metrics,
    compute_instance_metrics,
    compute_scene_graph_metrics
)
from recontext.core.camera import Camera
from recontext.language.scene_graph import SceneGraph
from recontext.utils.io_utils import ensure_dir
from recontext.config.paths import get_output_dir
from recontext.evaluation.visualize_results import visualize_metrics, generate_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RECONTEXT Evaluation Tool")
    
    # Input paths
    parser.add_argument("--prediction", required=True, help="Path to prediction directory")
    parser.add_argument("--ground_truth", required=True, help="Path to ground truth directory")
    parser.add_argument("--output", help="Path to save evaluation results")
    
    # Evaluation options
    parser.add_argument("--eval_geometry", action="store_true", help="Evaluate geometric reconstruction")
    parser.add_argument("--eval_semantics", action="store_true", help="Evaluate semantic understanding")
    parser.add_argument("--eval_scene_graph", action="store_true", help="Evaluate scene graph")
    parser.add_argument("--eval_all", action="store_true", help="Evaluate all aspects")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--report", action="store_true", help="Generate evaluation report")
    
    # Specific metrics options
    parser.add_argument("--pcd_voxel_size", type=float, default=0.01,
                      help="Voxel size for point cloud evaluation (m)")
    parser.add_argument("--max_distance", type=float, default=0.1,
                      help="Maximum distance threshold for precision/recall (m)")
    
    return parser.parse_args()

def load_prediction_data(prediction_dir: str) -> Dict:
    """Load prediction data from directory.
    
    Args:
        prediction_dir: Path to prediction directory
        
    Returns:
        Dictionary of prediction data
    """
    logger.info(f"Loading prediction data from {prediction_dir}")
    
    prediction = {}
    
    # Load point cloud if available
    pcd_paths = [
        os.path.join(prediction_dir, "pointcloud.ply"),
        os.path.join(prediction_dir, "dense_pointcloud.ply"),
        os.path.join(prediction_dir, "sparse_pointcloud.ply")
    ]
    
    for pcd_path in pcd_paths:
        if os.path.exists(pcd_path):
            try:
                pcd = o3d.io.read_point_cloud(pcd_path)
                if pcd.has_points():
                    prediction["pointcloud"] = pcd
                    logger.info(f"Loaded point cloud with {len(pcd.points)} points from {pcd_path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load point cloud from {pcd_path}: {e}")
    
    # Load mesh if available
    mesh_paths = [
        os.path.join(prediction_dir, "mesh.ply"),
        os.path.join(prediction_dir, "surface.ply"),
        os.path.join(prediction_dir, "reconstruction.ply")
    ]
    
    for mesh_path in mesh_paths:
        if os.path.exists(mesh_path):
            try:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                if mesh.has_triangles():
                    prediction["mesh"] = mesh
                    logger.info(f"Loaded mesh with {len(mesh.triangles)} triangles from {mesh_path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load mesh from {mesh_path}: {e}")
    
    # Load camera poses if available
    cameras_path = os.path.join(prediction_dir, "cameras.pkl")
    if os.path.exists(cameras_path):
        try:
            import pickle
            with open(cameras_path, 'rb') as f:
                cameras = pickle.load(f)
            prediction["cameras"] = cameras
            logger.info(f"Loaded {len(cameras)} camera poses from {cameras_path}")
        except Exception as e:
            logger.warning(f"Failed to load cameras from {cameras_path}: {e}")
    
    # Load semantic data if available
    point_labels_path = os.path.join(prediction_dir, "point_labels.npy")
    if os.path.exists(point_labels_path):
        try:
            point_labels = np.load(point_labels_path)
            prediction["point_labels"] = point_labels
            logger.info(f"Loaded {len(point_labels)} point labels from {point_labels_path}")
        except Exception as e:
            logger.warning(f"Failed to load point labels from {point_labels_path}: {e}")
    
    # Load scene graph if available
    scene_graph_path = os.path.join(prediction_dir, "scene_graph.pkl")
    if os.path.exists(scene_graph_path):
        try:
            scene_graph = SceneGraph.load(scene_graph_path)
            prediction["scene_graph"] = scene_graph
            logger.info(f"Loaded scene graph with {len(scene_graph.objects)} objects from {scene_graph_path}")
        except Exception as e:
            logger.warning(f"Failed to load scene graph from {scene_graph_path}: {e}")
    
    return prediction

def load_ground_truth_data(ground_truth_dir: str) -> Dict:
    """Load ground truth data from directory.
    
    Args:
        ground_truth_dir: Path to ground truth directory
        
    Returns:
        Dictionary of ground truth data
    """
    logger.info(f"Loading ground truth data from {ground_truth_dir}")
    
    ground_truth = {}
    
    # Load point cloud if available
    pcd_paths = [
        os.path.join(ground_truth_dir, "pointcloud.ply"),
        os.path.join(ground_truth_dir, "dense_pointcloud.ply"),
        os.path.join(ground_truth_dir, "sparse_pointcloud.ply")
    ]
    
    for pcd_path in pcd_paths:
        if os.path.exists(pcd_path):
            try:
                pcd = o3d.io.read_point_cloud(pcd_path)
                if pcd.has_points():
                    ground_truth["pointcloud"] = pcd
                    logger.info(f"Loaded ground truth point cloud with {len(pcd.points)} points from {pcd_path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load ground truth point cloud from {pcd_path}: {e}")
    
    # Load mesh if available
    mesh_paths = [
        os.path.join(ground_truth_dir, "mesh.ply"),
        os.path.join(ground_truth_dir, "surface.ply"),
        os.path.join(ground_truth_dir, "reconstruction.ply")
    ]
    
    for mesh_path in mesh_paths:
        if os.path.exists(mesh_path):
            try:
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                if mesh.has_triangles():
                    ground_truth["mesh"] = mesh
                    logger.info(f"Loaded ground truth mesh with {len(mesh.triangles)} triangles from {mesh_path}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load ground truth mesh from {mesh_path}: {e}")
    
    # Load camera poses if available
    cameras_path = os.path.join(ground_truth_dir, "cameras.pkl")
    if os.path.exists(cameras_path):
        try:
            import pickle
            with open(cameras_path, 'rb') as f:
                cameras = pickle.load(f)
            ground_truth["cameras"] = cameras
            logger.info(f"Loaded {len(cameras)} ground truth camera poses from {cameras_path}")
        except Exception as e:
            logger.warning(f"Failed to load ground truth cameras from {cameras_path}: {e}")
    
    # Load semantic data if available
    point_labels_path = os.path.join(ground_truth_dir, "point_labels.npy")
    if os.path.exists(point_labels_path):
        try:
            point_labels = np.load(point_labels_path)
            ground_truth["point_labels"] = point_labels
            logger.info(f"Loaded {len(point_labels)} ground truth point labels from {point_labels_path}")
        except Exception as e:
            logger.warning(f"Failed to load ground truth point labels from {point_labels_path}: {e}")
    
    # Load scene graph if available
    scene_graph_path = os.path.join(ground_truth_dir, "scene_graph.pkl")
    if os.path.exists(scene_graph_path):
        try:
            scene_graph = SceneGraph.load(scene_graph_path)
            ground_truth["scene_graph"] = scene_graph
            logger.info(f"Loaded ground truth scene graph with {len(scene_graph.objects)} objects from {scene_graph_path}")
        except Exception as e:
            logger.warning(f"Failed to load ground truth scene graph from {scene_graph_path}: {e}")
    
    return ground_truth

def evaluate_geometry(prediction: Dict, ground_truth: Dict, args) -> Dict:
    """Evaluate geometric reconstruction.
    
    Args:
        prediction: Dictionary of prediction data
        ground_truth: Dictionary of ground truth data
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating geometric reconstruction")
    
    geometry_metrics = {}
    
    # Evaluate point cloud if available
    if "pointcloud" in prediction and "pointcloud" in ground_truth:
        logger.info("Evaluating point cloud")
        try:
            pointcloud_metrics = compute_pointcloud_metrics(
                prediction["pointcloud"],
                ground_truth["pointcloud"],
                downsample=True,
                voxel_size=args.pcd_voxel_size,
                max_distance=args.max_distance
            )
            geometry_metrics["pointcloud"] = pointcloud_metrics
        except Exception as e:
            logger.error(f"Failed to compute point cloud metrics: {e}")
    
    # Evaluate mesh if available
    if "mesh" in prediction and "mesh" in ground_truth:
        logger.info("Evaluating mesh")
        try:
            mesh_metrics = compute_mesh_metrics(
                prediction["mesh"],
                ground_truth["mesh"],
                max_distance=args.max_distance
            )
            
            # Add normal consistency
            normal_consistency = compute_normal_consistency(
                prediction["mesh"],
                ground_truth["mesh"]
            )
            mesh_metrics["normal_consistency"] = normal_consistency
            
            geometry_metrics["mesh"] = mesh_metrics
        except Exception as e:
            logger.error(f"Failed to compute mesh metrics: {e}")
    
    # Evaluate camera poses if available
    if "cameras" in prediction and "cameras" in ground_truth:
        logger.info("Evaluating camera poses")
        try:
            camera_metrics = compute_camera_pose_metrics(
                prediction["cameras"],
                ground_truth["cameras"]
            )
            geometry_metrics["cameras"] = camera_metrics
        except Exception as e:
            logger.error(f"Failed to compute camera pose metrics: {e}")
    
    return geometry_metrics

def evaluate_semantics(prediction: Dict, ground_truth: Dict, args) -> Dict:
    """Evaluate semantic understanding.
    
    Args:
        prediction: Dictionary of prediction data
        ground_truth: Dictionary of ground truth data
        args: Command line arguments
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating semantic understanding")
    
    semantic_metrics = {}
    
    # Evaluate point labels if available
    if ("pointcloud" in prediction and "pointcloud" in ground_truth and
        "point_labels" in prediction and "point_labels" in ground_truth):
        logger.info("Evaluating semantic segmentation")
        try:
            segmentation_metrics = compute_semantic_metrics(
                prediction["pointcloud"],
                prediction["point_labels"],
                ground_truth["pointcloud"],
                ground_truth["point_labels"]
            )
            semantic_metrics["segmentation"] = segmentation_metrics
        except Exception as e:
            logger.error(f"Failed to compute semantic segmentation metrics: {e}")
    
    # Evaluate instance segmentation if available
    # This would require instance labels which might not be directly available
    
    # Evaluate scene graph if available and requested
    if args.eval_scene_graph and "scene_graph" in prediction and "scene_graph" in ground_truth:
        logger.info("Evaluating scene graph")
        try:
            scene_graph_metrics = compute_scene_graph_metrics(
                prediction["scene_graph"],
                ground_truth["scene_graph"]
            )
            semantic_metrics["scene_graph"] = scene_graph_metrics
        except Exception as e:
            logger.error(f"Failed to compute scene graph metrics: {e}")
    
    return semantic_metrics

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Get output directory
    output_dir = args.output
    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(get_output_dir(), "evaluation", timestamp)
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Load prediction and ground truth data
    prediction = load_prediction_data(args.prediction)
    ground_truth = load_ground_truth_data(args.ground_truth)
    
    # Check if we have loaded data
    if not prediction:
        logger.error("No prediction data loaded")
        return 1
    
    if not ground_truth:
        logger.error("No ground truth data loaded")
        return 1
    
    # Initialize metrics container
    metrics = {}
    
    # Evaluate geometry if requested
    if args.eval_geometry or args.eval_all:
        geometry_metrics = evaluate_geometry(prediction, ground_truth, args)
        metrics["geometry"] = geometry_metrics
    
    # Evaluate semantics if requested
    if args.eval_semantics or args.eval_all:
        semantic_metrics = evaluate_semantics(prediction, ground_truth, args)
        metrics["semantics"] = semantic_metrics
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(metrics), f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Print summary
    print("\nEvaluation Summary:")
    
    if "geometry" in metrics:
        print("\nGeometry Metrics:")
        
        if "pointcloud" in metrics["geometry"]:
            pcd_metrics = metrics["geometry"]["pointcloud"]
            print(f"  Point Cloud:")
            print(f"    Chamfer Distance: {pcd_metrics.get('chamfer_distance', 'N/A'):.6f}")
            print(f"    F1 Score: {pcd_metrics.get('f1_score', 'N/A'):.6f}")
            print(f"    Precision: {pcd_metrics.get('precision', 'N/A'):.6f}")
            print(f"    Recall: {pcd_metrics.get('recall', 'N/A'):.6f}")
        
        if "mesh" in metrics["geometry"]:
            mesh_metrics = metrics["geometry"]["mesh"]
            print(f"  Mesh:")
            print(f"    Chamfer Distance: {mesh_metrics.get('chamfer_distance', 'N/A'):.6f}")
            print(f"    F1 Score: {mesh_metrics.get('f1_score', 'N/A'):.6f}")
            print(f"    Normal Consistency: {mesh_metrics.get('normal_consistency', 'N/A'):.6f}")
        
        if "cameras" in metrics["geometry"]:
            cam_metrics = metrics["geometry"]["cameras"]
            print(f"  Camera Poses:")
            print(f"    Rotation Error (deg): {cam_metrics.get('rotation_error_mean', 'N/A'):.6f}")
            print(f"    Translation Error: {cam_metrics.get('translation_error_mean', 'N/A'):.6f}")
    
    if "semantics" in metrics:
        print("\nSemantic Metrics:")
        
        if "segmentation" in metrics["semantics"]:
            seg_metrics = metrics["semantics"]["segmentation"]
            print(f"  Semantic Segmentation:")
            print(f"    mIoU: {seg_metrics.get('miou', 'N/A'):.6f}")
            print(f"    Accuracy: {seg_metrics.get('accuracy', 'N/A'):.6f}")
        
        if "scene_graph" in metrics["semantics"]:
            sg_metrics = metrics["semantics"]["scene_graph"]
            print(f"  Scene Graph:")
            print(f"    Object Recall: {sg_metrics.get('object_recall', 'N/A'):.6f}")
            print(f"    Relationship Recall: {sg_metrics.get('relationship_recall', 'N/A'):.6f}")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualizations")
        vis_dir = os.path.join(output_dir, "visualizations")
        ensure_dir(vis_dir)
        
        visualize_metrics(metrics, vis_dir, prediction, ground_truth)
        logger.info(f"Visualizations saved to {vis_dir}")
    
    # Generate report if requested
    if args.report:
        logger.info("Generating evaluation report")
        report_path = os.path.join(output_dir, "report.html")
        
        generate_report(metrics, report_path, args.prediction, args.ground_truth)
        logger.info(f"Report saved to {report_path}")
    
    logger.info("Evaluation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())