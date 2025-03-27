#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RECONTEXT: 3D Scene Reconstruction with Semantic Understanding
Main entry point that orchestrates the full 3D reconstruction and semantic understanding pipeline.

Author: James Wei & Alex Johnson
Date: 2024-02-01
"""

import os
import sys
import argparse
import logging
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from recontext.core.reconstruction import ReconstructionPipeline
from recontext.semantics.instance_segmentation import InstanceSegmentor
from recontext.integration.label_projection import LabelProjector
from recontext.language.scene_graph import SceneGraph, create_scene_graph_from_labeled_pointcloud
from recontext.language.query_engine import QueryEngine
from recontext.visualization.interactive_viewer import InteractiveViewer
from recontext.utils.io_utils import load_images, ensure_dir
from recontext.config.paths import get_data_dir, get_output_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('recontext.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RECONTEXT: 3D Scene Reconstruction with Semantic Understanding")
    
    # Input options
    parser.add_argument("--image_dir", help="Directory containing input images")
    parser.add_argument("--config", default="config/default_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output_dir", help="Directory to save outputs")
    
    # Pipeline control
    parser.add_argument("--skip_reconstruction", action="store_true",
                       help="Skip 3D reconstruction step")
    parser.add_argument("--skip_segmentation", action="store_true",
                       help="Skip semantic segmentation step")
    parser.add_argument("--skip_integration", action="store_true",
                       help="Skip semantic integration step")
    parser.add_argument("--skip_scene_graph", action="store_true",
                       help="Skip scene graph generation step")
    
    # Reconstruction options
    parser.add_argument("--use_neural", action="store_true",
                       help="Use neural implicit surfaces")
    parser.add_argument("--quality", choices=["low", "medium", "high"],
                       default="medium", help="Reconstruction quality")
    
    # Segmentation options
    parser.add_argument("--segmentation_model", choices=["mask2former_coco", "mask2former_ade20k"],
                       default="mask2former_coco", help="Segmentation model to use")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for segmentation")
    
    # Integration options
    parser.add_argument("--consensus", choices=["majority", "weighted", "bayesian", "confidence", "ensemble"],
                       default="ensemble", help="Consensus strategy for label integration")
    
    # Visualization option
    parser.add_argument("--visualize", action="store_true",
                       help="Open interactive visualizer after processing")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode without GUI")
    
    # Load existing files instead of processing
    parser.add_argument("--load_pointcloud", help="Load existing point cloud file")
    parser.add_argument("--load_mesh", help="Load existing mesh file")
    parser.add_argument("--load_scene_graph", help="Load existing scene graph file")
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def run_reconstruction(image_dir, output_dir, config):
    """Run 3D reconstruction pipeline.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save outputs
        config: Configuration dictionary
        
    Returns:
        Dictionary containing reconstruction results
    """
    logger.info("Starting 3D reconstruction")
    start_time = time.time()
    
    # Create reconstruction pipeline
    pipeline = ReconstructionPipeline(config)
    
    # Get image paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Run reconstruction
    results = pipeline.run(image_paths, output_dir)
    
    logger.info(f"Reconstruction completed in {time.time() - start_time:.2f} seconds")
    
    return results

def run_segmentation(image_dir, output_dir, config):
    """Run semantic segmentation on input images.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save outputs
        config: Configuration dictionary
        
    Returns:
        List of segmentation results for each image
    """
    logger.info("Starting semantic segmentation")
    start_time = time.time()
    
    # Get image paths
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort for consistent ordering
    image_paths.sort()
    
    # Create segmentor
    segmentor = InstanceSegmentor(
        model_type=config['segmentation']['model_type'],
        confidence_threshold=config['segmentation']['confidence_threshold'],
        enable_clip=config['segmentation']['enable_clip']
    )
    
    # Process images
    segmentation_results = segmentor.process_batch(image_paths)
    
    # Save results
    import pickle
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, 'segmentation_results.pkl'), 'wb') as f:
        pickle.dump(segmentation_results, f)
    
    # Generate visualizations
    vis_dir = os.path.join(output_dir, 'segmentation_vis')
    ensure_dir(vis_dir)
    
    for i, (image_path, instances) in enumerate(zip(image_paths, segmentation_results)):
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        vis_image = segmentor.visualize_results(image, instances)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"seg_{i:04d}.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Segmentation completed in {time.time() - start_time:.2f} seconds")
    
    return segmentation_results

def run_integration(reconstruction_results, segmentation_results, camera_params, output_dir, config):
    """Run semantic integration to project 2D labels to 3D.
    
    Args:
        reconstruction_results: Reconstruction results
        segmentation_results: Segmentation results
        camera_params: Camera parameters
        output_dir: Directory to save outputs
        config: Configuration dictionary
        
    Returns:
        Tuple of (point_labels, point_colors, label_metadata)
    """
    logger.info("Starting semantic integration")
    start_time = time.time()
    
    # Get point cloud from reconstruction results
    pointcloud = reconstruction_results['dense_pointcloud']
    
    # Create label projector
    projector = LabelProjector(
        visibility_threshold=config['integration']['visibility_threshold'],
        min_views_per_point=config['integration']['min_views_per_point'],
        max_projection_distance=config['integration']['max_projection_distance'],
        use_weighted_consensus=config['integration']['use_weighted_consensus']
    )
    
    # Project labels
    point_labels, point_colors, metadata = projector.project_labels(
        pointcloud, camera_params, segmentation_results)
    
    # Save results
    ensure_dir(output_dir)
    np.save(os.path.join(output_dir, 'point_labels.npy'), point_labels)
    np.save(os.path.join(output_dir, 'point_colors.npy'), point_colors)
    
    import json
    with open(os.path.join(output_dir, 'label_metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    # Create visualization
    labeled_pcd = projector.visualize_labels(pointcloud, point_labels, metadata)
    
    # Save labeled point cloud
    o3d.io.write_point_cloud(os.path.join(output_dir, 'labeled_pointcloud.ply'), labeled_pcd)
    
    # Project to mesh if available
    if 'mesh' in reconstruction_results and reconstruction_results['mesh'] is not None:
        labeled_mesh = projector.project_to_mesh(
            reconstruction_results['mesh'], pointcloud, point_labels, point_colors)
        
        # Save labeled mesh
        o3d.io.write_triangle_mesh(os.path.join(output_dir, 'labeled_mesh.ply'), labeled_mesh)
    
    logger.info(f"Semantic integration completed in {time.time() - start_time:.2f} seconds")
    
    return point_labels, point_colors, metadata

def run_scene_graph_generation(pointcloud, point_labels, metadata, output_dir, config):
    """Generate scene graph from labeled point cloud.
    
    Args:
        pointcloud: Labeled point cloud
        point_labels: Point labels
        metadata: Label metadata
        output_dir: Directory to save outputs
        config: Configuration dictionary
        
    Returns:
        Scene graph
    """
    logger.info("Starting scene graph generation")
    start_time = time.time()
    
    # Create scene graph
    scene_graph = create_scene_graph_from_labeled_pointcloud(pointcloud, point_labels, metadata)
    
    # Infer relationships
    scene_graph.infer_spatial_relationships(
        distance_threshold=config['scene_graph']['distance_threshold'],
        min_confidence=config['scene_graph']['min_confidence']
    )
    
    scene_graph.infer_functional_relationships()
    
    # Compute statistics
    scene_graph.compute_scene_statistics()
    
    # Save scene graph
    ensure_dir(output_dir)
    scene_graph.save(os.path.join(output_dir, 'scene_graph.pkl'))
    
    # Generate visualization
    scene_graph.visualize(os.path.join(output_dir, 'scene_graph.png'))
    
    logger.info(f"Scene graph generation completed in {time.time() - start_time:.2f} seconds")
    
    return scene_graph

def open_interactive_viewer(pointcloud_path, mesh_path, scene_graph_path):
    """Open interactive viewer with loaded files.
    
    Args:
        pointcloud_path: Path to point cloud file
        mesh_path: Path to mesh file
        scene_graph_path: Path to scene graph file
    """
    # Check if running in GUI environment
    if 'DISPLAY' not in os.environ and not sys.platform.startswith('win') and not sys.platform.startswith('darwin'):
        logger.error("Cannot open interactive viewer: No display available")
        return
    
    try:
        # Start interactive viewer
        viewer = InteractiveViewer(
            pointcloud_path=pointcloud_path,
            mesh_path=mesh_path,
            scene_graph_path=scene_graph_path
        )
        
        # Show viewer
        viewer.show()
        
        # Start Qt application loop
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error opening interactive viewer: {str(e)}")
        raise

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.image_dir:
        config['data']['image_dir'] = args.image_dir
    
    if args.output_dir:
        config['output']['base_dir'] = args.output_dir
    
    if args.use_neural:
        config['reconstruction']['use_neural_implicit'] = True
    
    if args.quality:
        config['reconstruction']['reconstruction_quality'] = args.quality
    
    if args.segmentation_model:
        config['segmentation']['model_type'] = args.segmentation_model
    
    if args.confidence:
        config['segmentation']['confidence_threshold'] = args.confidence
    
    if args.consensus:
        config['integration']['consensus_strategy'] = args.consensus
    
    # Get directories
    image_dir = config['data']['image_dir']
    if not image_dir:
        image_dir = get_data_dir()
    
    output_base_dir = config['output']['base_dir']
    if not output_base_dir:
        output_base_dir = get_output_dir()
    
    # Create output directories
    reconstruction_dir = os.path.join(output_base_dir, 'reconstruction')
    segmentation_dir = os.path.join(output_base_dir, 'segmentation')
    integration_dir = os.path.join(output_base_dir, 'integration')
    scene_graph_dir = os.path.join(output_base_dir, 'scene_graph')
    
    for directory in [reconstruction_dir, segmentation_dir, integration_dir, scene_graph_dir]:
        ensure_dir(directory)
    
    # Initialize result containers
    reconstruction_results = None
    segmentation_results = None
    camera_params = None
    point_labels = None
    point_colors = None
    label_metadata = None
    scene_graph = None
    
    # Paths for visualization
    pointcloud_path = None
    mesh_path = None
    scene_graph_path = None
    
    # Check if we're loading existing files
    if args.load_pointcloud:
        logger.info(f"Loading point cloud from {args.load_pointcloud}")
        try:
            import open3d as o3d
            pointcloud = o3d.io.read_point_cloud(args.load_pointcloud)
            reconstruction_results = {'dense_pointcloud': pointcloud}
            pointcloud_path = args.load_pointcloud
        except Exception as e:
            logger.error(f"Error loading point cloud: {str(e)}")
    
    if args.load_mesh:
        logger.info(f"Loading mesh from {args.load_mesh}")
        try:
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(args.load_mesh)
            if reconstruction_results is None:
                reconstruction_results = {}
            reconstruction_results['mesh'] = mesh
            mesh_path = args.load_mesh
        except Exception as e:
            logger.error(f"Error loading mesh: {str(e)}")
    
    if args.load_scene_graph:
        logger.info(f"Loading scene graph from {args.load_scene_graph}")
        try:
            scene_graph = SceneGraph.load(args.load_scene_graph)
            scene_graph_path = args.load_scene_graph
        except Exception as e:
            logger.error(f"Error loading scene graph: {str(e)}")
    
    # Start pipeline
    try:
        # Step 1: 3D Reconstruction
        if not args.skip_reconstruction and reconstruction_results is None:
            reconstruction_results = run_reconstruction(
                image_dir, reconstruction_dir, config['reconstruction'])
            
            # Set paths for visualization
            pointcloud_path = os.path.join(reconstruction_dir, 'dense_pointcloud.ply')
            mesh_path = os.path.join(reconstruction_dir, 'mesh.ply')
            
            # Get camera parameters
            camera_params = reconstruction_results['cameras']
        
        # Step 2: Semantic Segmentation
        if not args.skip_segmentation and segmentation_results is None:
            segmentation_results = run_segmentation(
                image_dir, segmentation_dir, config['segmentation'])
        
        # Step 3: Semantic Integration
        if not args.skip_integration and point_labels is None and reconstruction_results is not None and segmentation_results is not None:
            point_labels, point_colors, label_metadata = run_integration(
                reconstruction_results, segmentation_results, camera_params,
                integration_dir, config['integration'])
        
        # Step 4: Scene Graph Generation
        if not args.skip_scene_graph and scene_graph is None and reconstruction_results is not None and point_labels is not None:
            scene_graph = run_scene_graph_generation(
                reconstruction_results['dense_pointcloud'], point_labels, label_metadata,
                scene_graph_dir, config['scene_graph'])
            
            # Set path for visualization
            scene_graph_path = os.path.join(scene_graph_dir, 'scene_graph.pkl')
        
        # Open interactive visualizer if requested
        if args.visualize and not args.headless:
            if pointcloud_path or mesh_path or scene_graph_path:
                open_interactive_viewer(pointcloud_path, mesh_path, scene_graph_path)
            else:
                logger.warning("No data available for visualization")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("Pipeline completed successfully")
    return 0

if __name__ == "__main__":
    # Import visualization modules only if needed
    if any(arg in sys.argv for arg in ['--visualize']):
        try:
            import cv2
            import open3d as o3d
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            logger.error("Visualization dependencies not installed")
            sys.exit(1)
    
    sys.exit(main())