#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic 3D reconstruction example script.

This example demonstrates the basic reconstruction pipeline in RECONTEXT,
including Structure-from-Motion (SfM), Multi-View Stereo (MVS), and mesh generation.

Author: James Wei
Date: 2024-01-30
Last modified: 2024-03-05
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
import logging
import time
from pathlib import Path

from recontext.core.reconstruction import reconstruct_scene, ReconstructionPipeline
from recontext.utils.io_utils import load_images, ensure_dir
from recontext.visualization.pointcloud_vis import visualize_pointcloud
from recontext.visualization.mesh_vis import visualize_mesh

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic 3D Reconstruction Example")
    
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", default="reconstruction_output", help="Output directory")
    parser.add_argument("--method", choices=["colmap", "sfm_mvs", "neural"], default="colmap",
                       help="Reconstruction method to use")
    parser.add_argument("--feature_type", choices=["sift", "superpoint"], default="sift",
                       help="Feature type for SfM")
    parser.add_argument("--mesh_quality", choices=["low", "medium", "high"], default="medium",
                       help="Mesh quality")
    parser.add_argument("--visualize", action="store_true", help="Visualize results")
    parser.add_argument("--no_filter", action="store_true", help="Disable outlier filtering")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    return parser.parse_args()

def main():
    """Run the basic reconstruction example."""
    # Parse command line arguments
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print configuration
    logger.info("Reconstruction configuration:")
    logger.info(f"  Image directory: {args.image_dir}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Method: {args.method}")
    logger.info(f"  Feature type: {args.feature_type}")
    logger.info(f"  Mesh quality: {args.mesh_quality}")
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Get image paths
    image_dir = args.image_dir
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        logger.error(f"No images found in {image_dir}")
        return 1
    
    logger.info(f"Found {len(image_paths)} images")
    
    # Configure reconstruction
    config = {
        'feature_type': args.feature_type,
        'reconstruction_quality': args.mesh_quality,
        'filter_outliers': not args.no_filter,
    }
    
    if args.method == "neural":
        config['use_neural_implicit'] = True
        config['use_colmap'] = False
    elif args.method == "sfm_mvs":
        config['use_colmap'] = False
    else:  # colmap
        config['use_colmap'] = True
    
    try:
        # Start timer
        start_time = time.time()
        
        # Create reconstruction pipeline
        pipeline = ReconstructionPipeline(config)
        
        # Run reconstruction
        results = pipeline.run(image_paths, args.output_dir)
        
        # Log timing
        elapsed_time = time.time() - start_time
        logger.info(f"Reconstruction completed in {elapsed_time:.2f} seconds")
        
        # Log results
        logger.info("Reconstruction results:")
        logger.info(f"  Cameras: {len(results['cameras'])}")
        logger.info(f"  Sparse points: {len(results['sparse_pointcloud'].points)}")
        logger.info(f"  Dense points: {len(results['dense_pointcloud'].points)}")
        logger.info(f"  Mesh triangles: {len(results['mesh'].triangles)}")
        
        # Visualize results if requested
        if args.visualize:
            logger.info("Visualizing results...")
            
            # Visualize sparse point cloud
            visualize_pointcloud(
                results['sparse_pointcloud'],
                title="Sparse Reconstruction"
            )
            
            # Visualize dense point cloud
            visualize_pointcloud(
                results['dense_pointcloud'],
                title="Dense Reconstruction"
            )
            
            # Visualize mesh
            visualize_mesh(
                results['mesh'],
                title="Reconstructed Mesh"
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())