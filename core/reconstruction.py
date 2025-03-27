#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Core reconstruction pipeline combining SfM, MVS and mesh generation.
Author: James Wei
"""

import os
import logging
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union

from recontext.core.feature_extraction import extract_features
from recontext.core.matching import match_features
from recontext.core.sfm import run_sfm
from recontext.core.pointcloud import filter_pointcloud, densify_pointcloud
from recontext.core.mesh import create_mesh, improve_mesh
from recontext.core.neural_implicit import optimize_neural_surface
from recontext.utils.colmap_utils import run_colmap, parse_colmap_output
from recontext.utils.io_utils import load_images, save_reconstruction
from recontext.config.paths import get_cache_dir

logger = logging.getLogger(__name__)

class ReconstructionPipeline:
    """Main reconstruction pipeline that combines various 3D reconstruction steps."""
    
    def __init__(self, config: Dict):
        """Initialize reconstruction pipeline with configuration.
        
        Args:
            config: Dictionary with configuration parameters
        """
        self.config = config
        self.cache_dir = get_cache_dir()
        self.use_neural_implicit = config.get('use_neural_implicit', False)
        self.use_colmap = config.get('use_colmap', True)
        
        # Default parameters
        self.default_params = {
            'feature_type': 'superpoint',  # 'sift' or 'superpoint'
            'matcher_type': 'superglue',  # 'mutual_nn' or 'superglue'
            'reconstruction_quality': 'medium',  # 'low', 'medium', 'high'
            'filter_outliers': True,
            'min_track_length': 3,
        }
        
        # Update defaults with provided config
        for k, v in self.default_params.items():
            if k not in self.config:
                self.config[k] = v
                
        # Results containers
        self.features = None
        self.matches = None
        self.cameras = None
        self.pointcloud = None
        self.dense_pointcloud = None
        self.mesh = None
    
    def run(self, image_paths: List[str], output_dir: str) -> Dict:
        """Run the complete reconstruction pipeline.
        
        Args:
            image_paths: List of paths to input images
            output_dir: Directory to save reconstruction outputs
            
        Returns:
            Dict containing results including cameras, pointcloud, and mesh
        """
        logger.info(f"Starting reconstruction with {len(image_paths)} images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        images = load_images(image_paths)
        logger.info(f"Loaded {len(images)} images")
        
        if self.use_colmap:
            logger.info("Using COLMAP for reconstruction")
            colmap_dir = os.path.join(self.cache_dir, 'colmap_output')
            os.makedirs(colmap_dir, exist_ok=True)
            
            # Run COLMAP
            run_colmap(image_paths, colmap_dir, self.config['reconstruction_quality'])
            
            # Parse COLMAP output
            self.cameras, self.pointcloud = parse_colmap_output(colmap_dir)
        else:
            # Extract features
            logger.info(f"Extracting features using {self.config['feature_type']}")
            self.features = extract_features(images, feature_type=self.config['feature_type'])
            
            # Match features
            logger.info(f"Matching features using {self.config['matcher_type']}")
            self.matches = match_features(self.features, matcher_type=self.config['matcher_type'])
            
            # Structure from Motion
            logger.info("Running structure from motion")
            self.cameras, self.pointcloud = run_sfm(
                images, 
                self.features, 
                self.matches, 
                min_track_length=self.config['min_track_length']
            )
        
        logger.info(f"SfM complete: {len(self.cameras)} cameras, {len(self.pointcloud.points)} points")
        
        if self.config['filter_outliers']:
            logger.info("Filtering outliers from sparse pointcloud")
            self.pointcloud = filter_pointcloud(self.pointcloud)
            logger.info(f"After filtering: {len(self.pointcloud.points)} points")
        
        # MVS: Densify pointcloud
        logger.info("Densifying pointcloud")
        self.dense_pointcloud = densify_pointcloud(images, self.cameras, self.pointcloud)
        logger.info(f"Densification complete: {len(self.dense_pointcloud.points)} points")
        
        # Mesh generation
        if self.use_neural_implicit:
            logger.info("Generating mesh using neural implicit surfaces")
            self.mesh = optimize_neural_surface(self.dense_pointcloud, images, self.cameras)
            logger.info(f"Mesh created with {len(self.mesh.triangles)} triangles")
        else:
            logger.info("Generating mesh using traditional methods")
            self.mesh = create_mesh(self.dense_pointcloud)
            logger.info(f"Initial mesh created with {len(self.mesh.triangles)} triangles")
            
            logger.info("Improving mesh quality")
            self.mesh = improve_mesh(self.mesh)
            logger.info(f"Final mesh has {len(self.mesh.triangles)} triangles")
        
        # Save results
        results = {
            'cameras': self.cameras,
            'sparse_pointcloud': self.pointcloud,
            'dense_pointcloud': self.dense_pointcloud,
            'mesh': self.mesh
        }
        save_reconstruction(results, output_dir)
        logger.info(f"Reconstruction saved to {output_dir}")
        
        return results
    
    @staticmethod
    def visualize_result(mesh: o3d.geometry.TriangleMesh, 
                        pointcloud: Optional[o3d.geometry.PointCloud] = None) -> None:
        """Visualize reconstruction result.
        
        Args:
            mesh: Triangle mesh to visualize
            pointcloud: Point cloud to visualize alongside mesh (optional)
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        if pointcloud is not None:
            vis.add_geometry(pointcloud)
        
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()


def reconstruct_scene(image_dir: str, output_dir: str, config: Optional[Dict] = None) -> Dict:
    """Convenience function to reconstruct a scene from images in a directory.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save reconstruction outputs
        config: Optional configuration dictionary
    
    Returns:
        Dict containing reconstruction results
    """
    if config is None:
        config = {}
    
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    pipeline = ReconstructionPipeline(config)
    results = pipeline.run(image_paths, output_dir)
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run 3D reconstruction pipeline")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save reconstruction outputs")
    parser.add_argument("--use_neural", action="store_true", help="Use neural implicit surfaces")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium", 
                       help="Reconstruction quality")
    
    args = parser.parse_args()
    
    config = {
        'use_neural_implicit': args.use_neural,
        'reconstruction_quality': args.quality
    }
    
    results = reconstruct_scene(args.image_dir, args.output_dir, config)
    
    print(f"Reconstruction complete with {len(results['mesh'].triangles)} mesh triangles")