#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for visualizing semantic segmentation of 3D scenes.

This demonstrates how to load and visualize semantically labeled 3D data,
including point clouds and meshes, with color-coded semantic labels.

Author: Alex Johnson
Date: 2024-02-20
Last modified: 2024-03-18
"""

import os
import sys
import argparse
import logging
import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recontext.semantics.label_manager import LabelManager
from recontext.utils.io_utils import ensure_dir
from recontext.visualization.pointcloud_vis import visualize_labeled_pointcloud
from recontext.visualization.mesh_vis import visualize_labeled_mesh
from recontext.visualization.scene_graph_vis import visualize_scene_graph
from recontext.language.scene_graph import SceneGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RECONTEXT Semantic Visualization Example")
    
    parser.add_argument("--pointcloud", help="Path to point cloud file (.ply)")
    parser.add_argument("--point_labels", help="Path to point labels file (.npy)")
    parser.add_argument("--mesh", help="Path to mesh file (.ply)")
    parser.add_argument("--mesh_labels", help="Path to mesh labels file (.npy or .json)")
    parser.add_argument("--scene_graph", help="Path to scene graph file (.pkl)")
    parser.add_argument("--metadata", help="Path to label metadata file (.json)")
    parser.add_argument("--output", help="Output directory for visualizations")
    parser.add_argument("--format", choices=["png", "html", "both"], default="both",
                       help="Output format for visualizations")
    parser.add_argument("--interactive", action="store_true", 
                       help="Enable interactive visualization (if available)")
    parser.add_argument("--filter_labels", nargs='*', 
                       help="Only show specific labels (space-separated)")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                       help="Minimum confidence for scene graph objects")
    
    return parser.parse_args()

def load_label_metadata(metadata_path=None):
    """Load label metadata from file or use default."""
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded label metadata from {metadata_path}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
    
    # Use default label manager
    label_manager = LabelManager()
    logger.info("Using default label metadata")
    return label_manager.get_default_metadata()

def visualize_semantics(args):
    """Visualize semantic segmentation of 3D data."""
    # Prepare output directory
    output_dir = args.output
    if output_dir:
        ensure_dir(output_dir)
    
    # Load label metadata
    metadata = load_label_metadata(args.metadata)
    
    # Create label filter if specified
    label_filter = set(args.filter_labels) if args.filter_labels else None
    
    # Visualize labeled point cloud if provided
    if args.pointcloud and args.point_labels:
        logger.info(f"Visualizing labeled point cloud")
        
        # Load point cloud
        try:
            pointcloud = o3d.io.read_point_cloud(args.pointcloud)
            point_labels = np.load(args.point_labels)
            
            logger.info(f"Loaded point cloud with {len(pointcloud.points)} points and {len(point_labels)} labels")
            
            # Visualize
            if args.interactive:
                # Interactive visualization
                visualize_labeled_pointcloud(pointcloud, point_labels, metadata, 
                                          interactive=True, label_filter=label_filter)
            
            if output_dir:
                # Static visualization
                if args.format in ["png", "both"]:
                    img_path = os.path.join(output_dir, "labeled_pointcloud.png")
                    visualize_labeled_pointcloud(pointcloud, point_labels, metadata, 
                                             output_path=img_path, label_filter=label_filter)
                    logger.info(f"Saved point cloud visualization to {img_path}")
                
                if args.format in ["html", "both"]:
                    html_path = os.path.join(output_dir, "labeled_pointcloud.html")
                    visualize_labeled_pointcloud(pointcloud, point_labels, metadata, 
                                             output_path=html_path, format="html", 
                                             label_filter=label_filter)
                    logger.info(f"Saved interactive point cloud visualization to {html_path}")
        
        except Exception as e:
            logger.error(f"Failed to visualize labeled point cloud: {e}")
    
    # Visualize labeled mesh if provided
    if args.mesh:
        logger.info(f"Visualizing mesh")
        
        try:
            # Load mesh
            mesh = o3d.io.read_triangle_mesh(args.mesh)
            
            # Load mesh labels if provided
            mesh_labels = None
            if args.mesh_labels:
                if args.mesh_labels.endswith(".npy"):
                    mesh_labels = np.load(args.mesh_labels)
                elif args.mesh_labels.endswith(".json"):
                    with open(args.mesh_labels, 'r') as f:
                        mesh_labels = json.load(f)
            
            logger.info(f"Loaded mesh with {len(mesh.triangles)} triangles")
            
            # Visualize
            if args.interactive:
                # Interactive visualization
                visualize_labeled_mesh(mesh, mesh_labels, metadata, 
                                     interactive=True, label_filter=label_filter)
            
            if output_dir:
                # Static visualization
                if args.format in ["png", "both"]:
                    img_path = os.path.join(output_dir, "labeled_mesh.png")
                    visualize_labeled_mesh(mesh, mesh_labels, metadata, 
                                        output_path=img_path, label_filter=label_filter)
                    logger.info(f"Saved mesh visualization to {img_path}")
                
                if args.format in ["html", "both"]:
                    html_path = os.path.join(output_dir, "labeled_mesh.html")
                    visualize_labeled_mesh(mesh, mesh_labels, metadata, 
                                        output_path=html_path, format="html",
                                        label_filter=label_filter)
                    logger.info(f"Saved interactive mesh visualization to {html_path}")
        
        except Exception as e:
            logger.error(f"Failed to visualize mesh: {e}")
    
    # Visualize scene graph if provided
    if args.scene_graph:
        logger.info(f"Visualizing scene graph")
        
        try:
            # Load scene graph
            scene_graph = SceneGraph.load(args.scene_graph)
            logger.info(f"Loaded scene graph with {len(scene_graph.objects)} objects and {len(scene_graph.relationships)} relationships")
            
            # Filter by confidence if needed
            if args.min_confidence > 0:
                filtered_objects = {}
                for obj_id, obj in scene_graph.objects.items():
                    if obj.confidence >= args.min_confidence:
                        filtered_objects[obj_id] = obj
                
                # Update scene graph
                original_count = len(scene_graph.objects)
                scene_graph.objects = filtered_objects
                logger.info(f"Filtered scene graph to {len(scene_graph.objects)} objects (from {original_count}) with confidence >= {args.min_confidence}")
            
            # Visualize
            if args.interactive:
                # Call the interactive visualizer
                try:
                    from recontext.visualization.interactive_viewer import InteractiveViewer
                    viewer = InteractiveViewer(
                        pointcloud_path=args.pointcloud,
                        mesh_path=args.mesh,
                        scene_graph_path=args.scene_graph
                    )
                    viewer.show()
                except Exception as e:
                    logger.warning(f"Could not launch interactive viewer: {e}")
                    # Fall back to static visualization
                    visualize_scene_graph(scene_graph, interactive=True)
            
            if output_dir:
                # Static visualization
                if args.format in ["png", "both"]:
                    img_path = os.path.join(output_dir, "scene_graph.png")
                    visualize_scene_graph(scene_graph, output_path=img_path)
                    logger.info(f"Saved scene graph visualization to {img_path}")
                
                # Also save a schematic view of the scene graph
                schematic_path = os.path.join(output_dir, "scene_graph_schematic.png")
                scene_graph.visualize(schematic_path)
                logger.info(f"Saved scene graph schematic to {schematic_path}")
                
                # Object distribution visualization
                objects_path = os.path.join(output_dir, "object_distribution.png")
                visualize_object_distribution(scene_graph, objects_path)
                logger.info(f"Saved object distribution visualization to {objects_path}")
                
                # Relationship distribution visualization
                relationships_path = os.path.join(output_dir, "relationship_distribution.png")
                visualize_relationship_distribution(scene_graph, relationships_path)
                logger.info(f"Saved relationship distribution visualization to {relationships_path}")
        
        except Exception as e:
            logger.error(f"Failed to visualize scene graph: {e}")
    
    logger.info("Visualization complete")

def visualize_object_distribution(scene_graph, output_path):
    """Visualize distribution of object types in scene graph."""
    # Count object types
    object_counts = {}
    for obj in scene_graph.objects.values():
        object_counts[obj.label] = object_counts.get(obj.label, 0) + 1
    
    # Sort by count
    sorted_counts = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_counts]
    counts = [item[1] for item in sorted_counts]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Object Type')
    plt.ylabel('Count')
    plt.title('Distribution of Object Types in Scene')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def visualize_relationship_distribution(scene_graph, output_path):
    """Visualize distribution of relationship types in scene graph."""
    # Count relationship types
    relation_counts = {}
    for rel in scene_graph.relationships.values():
        relation_counts[rel.type] = relation_counts.get(rel.type, 0) + 1
    
    # Sort by count
    sorted_counts = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_counts]
    counts = [item[1] for item in sorted_counts]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts, color='salmon')
    plt.xlabel('Relationship Type')
    plt.ylabel('Count')
    plt.title('Distribution of Relationship Types in Scene')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check if any visualization inputs are provided
    if not any([args.pointcloud, args.mesh, args.scene_graph]):
        logger.error("No visualization inputs provided. Please specify at least one of: --pointcloud, --mesh, or --scene_graph")
        return 1
    
    # Run visualization
    visualize_semantics(args)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())