#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline methods for 3D reconstruction and scene understanding.

This module implements baseline methods that serve as a reference point
for evaluating the performance of the RECONTEXT system.

Author: Sarah Li
Date: 2024-01-22
Last modified: 2024-03-14
"""

import os
import numpy as np
import open3d as o3d
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json

# Try to import COLMAP to handle baselines
try:
    import pycolmap
    has_colmap = True
except ImportError:
    has_colmap = False
    logging.warning("pycolmap not found. COLMAP baseline functions will be limited.")

from recontext.core.pointcloud import filter_pointcloud, compute_normals
from recontext.core.mesh import create_mesh
from recontext.utils.colmap_utils import run_colmap, parse_colmap_output

logger = logging.getLogger(__name__)

class BaselineMethod:
    """Base class for baseline reconstruction methods."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        """Initialize baseline method.
        
        Args:
            name: Name of the baseline method
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.results = {}
        self.metrics = {}
        self.runtime = 0.0
    
    def run(self, 
            image_paths: List[str], 
            output_dir: str, 
            **kwargs) -> Dict:
        """Run baseline reconstruction method.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of results
        """
        raise NotImplementedError("Subclasses must implement run")
    
    def evaluate(self, ground_truth: Dict) -> Dict:
        """Evaluate baseline method against ground truth.
        
        Args:
            ground_truth: Ground truth data
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement evaluate")
    
    def save_results(self, output_dir: str) -> None:
        """Save baseline results.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        if self.metrics:
            metrics_path = os.path.join(output_dir, f"{self.name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        
        # Save runtime
        runtime_path = os.path.join(output_dir, f"{self.name}_runtime.txt")
        with open(runtime_path, 'w') as f:
            f.write(f"Runtime: {self.runtime:.2f} seconds\n")


class COLMAPBaseline(BaselineMethod):
    """COLMAP baseline for 3D reconstruction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize COLMAP baseline.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("colmap", config)
        
        # Default configuration
        default_config = {
            'feature_extractor': 'sift',
            'matcher': 'exhaustive',
            'reconstruction_quality': 'medium',
            'mesh_method': 'poisson',
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
            
        self.config = default_config
    
    def run(self, 
            image_paths: List[str], 
            output_dir: str, 
            **kwargs) -> Dict:
        """Run COLMAP baseline reconstruction.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of results
        """
        logger.info(f"Running COLMAP baseline with {len(image_paths)} images")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Start timer
        start_time = time.time()
        
        # Run COLMAP
        colmap_dir = os.path.join(output_dir, 'colmap')
        os.makedirs(colmap_dir, exist_ok=True)
        
        success = run_colmap(
            image_paths, 
            colmap_dir, 
            self.config['reconstruction_quality']
        )
        
        if not success:
            logger.error("COLMAP reconstruction failed")
            self.runtime = time.time() - start_time
            return {'success': False}
        
        # Parse COLMAP output
        cameras, pointcloud = parse_colmap_output(colmap_dir)
        
        # Filter point cloud
        filtered_pointcloud = filter_pointcloud(pointcloud)
        
        # Create mesh
        mesh = None
        try:
            # Ensure normals are available
            if not filtered_pointcloud.has_normals():
                filtered_pointcloud = compute_normals(filtered_pointcloud)
                
            mesh = create_mesh(
                filtered_pointcloud, 
                method=self.config['mesh_method']
            )
        except Exception as e:
            logger.error(f"Mesh creation failed: {e}")
        
        # Save results
        o3d.io.write_point_cloud(
            os.path.join(output_dir, 'colmap_pointcloud.ply'), 
            filtered_pointcloud
        )
        
        if mesh is not None:
            o3d.io.write_triangle_mesh(
                os.path.join(output_dir, 'colmap_mesh.ply'), 
                mesh
            )
        
        # Record runtime
        self.runtime = time.time() - start_time
        logger.info(f"COLMAP baseline completed in {self.runtime:.2f} seconds")
        
        # Store results
        self.results = {
            'success': True,
            'cameras': cameras,
            'pointcloud': filtered_pointcloud,
            'mesh': mesh,
            'runtime': self.runtime
        }
        
        return self.results
    
    def evaluate(self, ground_truth: Dict) -> Dict:
        """Evaluate COLMAP baseline against ground truth.
        
        Args:
            ground_truth: Ground truth data
            
        Returns:
            Dictionary of evaluation metrics
        """
        from recontext.evaluation.geometry_metrics import (
            compute_pointcloud_metrics,
            compute_mesh_metrics
        )
        
        metrics = {}
        
        # Evaluate point cloud if available
        if 'pointcloud' in self.results and 'gt_pointcloud' in ground_truth:
            pointcloud_metrics = compute_pointcloud_metrics(
                self.results['pointcloud'],
                ground_truth['gt_pointcloud']
            )
            metrics['pointcloud'] = pointcloud_metrics
        
        # Evaluate mesh if available
        if 'mesh' in self.results and self.results['mesh'] is not None and 'gt_mesh' in ground_truth:
            mesh_metrics = compute_mesh_metrics(
                self.results['mesh'],
                ground_truth['gt_mesh']
            )
            metrics['mesh'] = mesh_metrics
        
        # Store metrics
        self.metrics = metrics
        
        return metrics


class MVSNetBaseline(BaselineMethod):
    """MVSNet baseline for dense reconstruction."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize MVSNet baseline.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("mvsnet", config)
        
        # Default configuration
        default_config = {
            'model_weights': 'mvsnet_weights.pth',
            'num_views': 5,
            'depth_max': 256,
            'depth_interval': 2.0,
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
            
        self.config = default_config
        
        # Check if PyTorch is available
        try:
            import torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load model weights
            self._load_model()
            
        except ImportError:
            logger.error("PyTorch not found. MVSNet baseline requires PyTorch.")
            self.model = None
    
    def _load_model(self):
        """Load MVSNet model weights."""
        try:
            import torch
            from recontext.core.neural_mvsnet import MVSNet
            
            # Initialize model
            self.model = MVSNet(
                depth_max=self.config['depth_max'],
                depth_interval=self.config['depth_interval']
            )
            
            # Load weights
            weights_path = self.config['model_weights']
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"MVSNet model loaded from {weights_path}")
            else:
                logger.warning(f"Model weights not found at {weights_path}")
                self.model = None
                
        except (ImportError, ModuleNotFoundError):
            logger.error("Failed to load MVSNet model")
            self.model = None
    
    def run(self, 
            image_paths: List[str], 
            output_dir: str, 
            **kwargs) -> Dict:
        """Run MVSNet baseline reconstruction.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of results
        """
        logger.info(f"Running MVSNet baseline with {len(image_paths)} images")
        
        if self.model is None:
            logger.error("MVSNet model not initialized")
            return {'success': False}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get camera poses (typically would be from COLMAP)
        if 'cameras' not in kwargs:
            logger.error("Camera poses are required for MVSNet")
            return {'success': False}
            
        cameras = kwargs['cameras']
        
        # Start timer
        start_time = time.time()
        
        # TODO: Implement MVSNet depth estimation and fusion
        # This is a placeholder - actual implementation would run the MVSNet model
        
        logger.warning("MVSNet implementation is incomplete")
        
        # Record runtime
        self.runtime = time.time() - start_time
        logger.info(f"MVSNet baseline completed in {self.runtime:.2f} seconds")
        
        # Store results
        self.results = {
            'success': False,  # Change to True when implemented
            'runtime': self.runtime
        }
        
        return self.results
    
    def evaluate(self, ground_truth: Dict) -> Dict:
        """Evaluate MVSNet baseline against ground truth.
        
        Args:
            ground_truth: Ground truth data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Placeholder for evaluation
        metrics = {}
        
        # Store metrics
        self.metrics = metrics
        
        return metrics


class Mask2FormerBaseline(BaselineMethod):
    """Mask2Former baseline for semantic segmentation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Mask2Former baseline.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__("mask2former", config)
        
        # Default configuration
        default_config = {
            'model_type': 'mask2former_coco',
            'confidence_threshold': 0.5,
            'enable_clip': False
        }
        
        # Update with provided config
        if config:
            default_config.update(config)
            
        self.config = default_config
        
        # Initialize segmentation model
        try:
            from recontext.semantics.instance_segmentation import InstanceSegmentor
            
            self.segmentor = InstanceSegmentor(
                model_type=self.config['model_type'],
                confidence_threshold=self.config['confidence_threshold'],
                enable_clip=self.config['enable_clip']
            )
            
        except (ImportError, ModuleNotFoundError):
            logger.error("Failed to initialize Mask2Former")
            self.segmentor = None
    
    def run(self, 
            image_paths: List[str], 
            output_dir: str, 
            **kwargs) -> Dict:
        """Run Mask2Former baseline segmentation.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save output
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of results
        """
        logger.info(f"Running Mask2Former baseline with {len(image_paths)} images")
        
        if self.segmentor is None:
            logger.error("Mask2Former not initialized")
            return {'success': False}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Start timer
        start_time = time.time()
        
        # Process images
        segmentation_results = self.segmentor.process_batch(image_paths)
        
        # Save results
        import pickle
        with open(os.path.join(output_dir, 'segmentation_results.pkl'), 'wb') as f:
            pickle.dump(segmentation_results, f)
        
        # Generate and save visualizations
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        import cv2
        for i, (path, instances) in enumerate(zip(image_paths, segmentation_results)):
            # Read image
            image = cv2.imread(path)
            if image is None:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create visualization
            vis_image = self.segmentor.visualize_results(image, instances)
            
            # Save visualization
            filename = os.path.basename(path)
            save_path = os.path.join(vis_dir, f"seg_{filename}")
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Record runtime
        self.runtime = time.time() - start_time
        logger.info(f"Mask2Former baseline completed in {self.runtime:.2f} seconds")
        
        # Store results
        self.results = {
            'success': True,
            'segmentation_results': segmentation_results,
            'runtime': self.runtime
        }
        
        return self.results
    
    def evaluate(self, ground_truth: Dict) -> Dict:
        """Evaluate Mask2Former baseline against ground truth.
        
        Args:
            ground_truth: Ground truth data
            
        Returns:
            Dictionary of evaluation metrics
        """
        from recontext.evaluation.semantic_metrics import compute_segmentation_metrics
        
        metrics = {}
        
        # Evaluate segmentation if available
        if 'segmentation_results' in self.results and 'gt_segmentation' in ground_truth:
            sem_metrics = compute_segmentation_metrics(
                self.results['segmentation_results'],
                ground_truth['gt_segmentation']
            )
            metrics['segmentation'] = sem_metrics
        
        # Store metrics
        self.metrics = metrics
        
        return metrics


def run_all_baselines(image_paths: List[str], 
                     output_dir: str,
                     ground_truth: Optional[Dict] = None,
                     **kwargs) -> Dict[str, Dict]:
    """Run all baseline methods and evaluate them.
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save output
        ground_truth: Optional ground truth data for evaluation
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary of baseline results and metrics
    """
    logger.info(f"Running all baselines with {len(image_paths)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize baselines
    baselines = {
        'colmap': COLMAPBaseline(kwargs.get('colmap_config')),
        'mask2former': Mask2FormerBaseline(kwargs.get('mask2former_config')),
    }
    
    # Add MVSNet if requested
    if kwargs.get('run_mvsnet', False):
        baselines['mvsnet'] = MVSNetBaseline(kwargs.get('mvsnet_config'))
    
    # Run baselines
    results = {}
    for name, baseline in baselines.items():
        logger.info(f"Running {name} baseline")
        baseline_dir = os.path.join(output_dir, name)
        os.makedirs(baseline_dir, exist_ok=True)
        
        try:
            # Run baseline
            baseline_results = baseline.run(image_paths, baseline_dir, **kwargs)
            
            # Evaluate if ground truth is available
            if ground_truth is not None:
                metrics = baseline.evaluate(ground_truth)
                baseline.save_results(baseline_dir)
            else:
                metrics = {}
            
            results[name] = {
                'results': baseline_results,
                'metrics': metrics,
                'runtime': baseline.runtime
            }
            
        except Exception as e:
            logger.error(f"Error running {name} baseline: {e}")
            results[name] = {
                'results': {'success': False, 'error': str(e)},
                'metrics': {},
                'runtime': 0.0
            }
    
    # Save summary
    summary = {name: {
        'success': res['results'].get('success', False),
        'runtime': res['runtime'],
        'metrics': {k: v.get('mean', 0.0) if isinstance(v, dict) and 'mean' in v else v 
                   for k, v in res['metrics'].items()} if res['metrics'] else {}
    } for name, res in results.items()}
    
    with open(os.path.join(output_dir, 'baselines_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("All baselines completed")
    
    return results


def compare_with_recontext(recontext_results: Dict, 
                          baseline_results: Dict[str, Dict],
                          output_dir: str) -> Dict:
    """Compare RECONTEXT results with baselines.
    
    Args:
        recontext_results: RECONTEXT results
        baseline_results: Baseline results
        output_dir: Directory to save comparison
        
    Returns:
        Dictionary of comparison metrics
    """
    from recontext.evaluation.visualize_results import generate_comparison_charts
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    recontext_metrics = recontext_results.get('metrics', {})
    baseline_metrics = {name: res['metrics'] for name, res in baseline_results.items()}
    
    # Extract runtimes
    recontext_runtime = recontext_results.get('runtime', 0.0)
    baseline_runtimes = {name: res['runtime'] for name, res in baseline_results.items()}
    
    # Compile comparison
    comparison = {
        'metrics': {
            'recontext': recontext_metrics,
            **baseline_metrics
        },
        'runtime': {
            'recontext': recontext_runtime,
            **baseline_runtimes
        }
    }
    
    # Generate comparison charts
    chart_path = os.path.join(output_dir, 'comparison_charts.png')
    generate_comparison_charts(comparison, chart_path)
    
    # Save comparison data
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {output_dir}")
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run baseline methods")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save output")
    parser.add_argument("--method", choices=["colmap", "mask2former", "all"],
                       default="all", help="Baseline method to run")
    parser.add_argument("--gt_dir", help="Directory containing ground truth data")
    
    args = parser.parse_args()
    
    # Get image paths
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_paths.extend([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                          if f.lower().endswith(ext)])
    
    # Sort image paths
    image_paths.sort()
    
    if not image_paths:
        logger.error(f"No images found in {args.image_dir}")
        import sys
        sys.exit(1)
    
    # Load ground truth if available
    ground_truth = None
    if args.gt_dir and os.path.exists(args.gt_dir):
        ground_truth = {}
        
        # Load ground truth point cloud if available
        gt_pcd_path = os.path.join(args.gt_dir, 'gt_pointcloud.ply')
        if os.path.exists(gt_pcd_path):
            ground_truth['gt_pointcloud'] = o3d.io.read_point_cloud(gt_pcd_path)
        
        # Load ground truth mesh if available
        gt_mesh_path = os.path.join(args.gt_dir, 'gt_mesh.ply')
        if os.path.exists(gt_mesh_path):
            ground_truth['gt_mesh'] = o3d.io.read_triangle_mesh(gt_mesh_path)
        
        # Load ground truth segmentation if available
        gt_seg_path = os.path.join(args.gt_dir, 'gt_segmentation.pkl')
        if os.path.exists(gt_seg_path):
            import pickle
            with open(gt_seg_path, 'rb') as f:
                ground_truth['gt_segmentation'] = pickle.load(f)
    
    # Run baselines
    if args.method == "colmap":
        # Run only COLMAP
        baseline = COLMAPBaseline()
        result = baseline.run(image_paths, args.output_dir)
        
        if ground_truth:
            metrics = baseline.evaluate(ground_truth)
            baseline.save_results(args.output_dir)
            
        logger.info(f"COLMAP baseline completed in {baseline.runtime:.2f} seconds")
        
    elif args.method == "mask2former":
        # Run only Mask2Former
        baseline = Mask2FormerBaseline()
        result = baseline.run(image_paths, args.output_dir)
        
        if ground_truth:
            metrics = baseline.evaluate(ground_truth)
            baseline.save_results(args.output_dir)
            
        logger.info(f"Mask2Former baseline completed in {baseline.runtime:.2f} seconds")
        
    else:  # all
        # Run all baselines
        results = run_all_baselines(image_paths, args.output_dir, ground_truth)
        
        # Print summary
        for name, res in results.items():
            success = res['results'].get('success', False)
            runtime = res['runtime']
            logger.info(f"{name}: {'Success' if success else 'Failed'}, Runtime: {runtime:.2f}s")