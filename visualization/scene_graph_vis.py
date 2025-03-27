#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scene graph visualization module.

This module provides tools for visualizing 3D scene graphs, including
object relationships, spatial layout, and interactive exploration.

Author: Michael Zhang
Date: 2024-02-15
Last modified: 2024-03-10
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import logging
import os

from recontext.language.scene_graph import SceneGraph, Object3D, Relationship
from recontext.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

class SceneGraphVisualizer:
    """Visualizer for 3D scene graphs."""
    
    def __init__(self, 
                 scene_graph: SceneGraph,
                 window_name: str = "Scene Graph Visualization"):
        """Initialize scene graph visualizer.
        
        Args:
            scene_graph: Scene graph to visualize
            window_name: Window title
        """
        self.scene_graph = scene_graph
        self.window_name = window_name
        
        # Default view settings
        self.background_color = [0.1, 0.1, 0.1]  # Dark gray
        self.point_size = 2.0
        self.show_coordinate_frame = True
        
        # Visualization options
        self.show_object_labels = True
        self.show_bounding_boxes = True
        self.show_relationships = True
        self.relationship_types_to_show = set()  # Empty means show all
        self.highlight_objects = set()  # Object IDs to highlight
        
        # Colors
        self.highlight_color = [1.0, 0.5, 0.0]  # Orange
        self.relationship_colors = {
            'above': [0.0, 0.8, 1.0],
            'below': [0.0, 0.6, 0.8],
            'left_of': [0.8, 0.0, 0.0],
            'right_of': [1.0, 0.0, 0.0],
            'in_front_of': [0.0, 1.0, 0.0],
            'behind': [0.0, 0.8, 0.0],
            'inside': [0.8, 0.0, 0.8],
            'contains': [1.0, 0.0, 1.0],
            'on_top_of': [1.0, 0.8, 0.0],
            'under': [0.8, 0.6, 0.0],
            'next_to': [0.5, 0.5, 0.5],
            'near': [0.7, 0.7, 0.7],
            'default': [0.0, 0.0, 1.0]  # Default blue
        }
    
    def visualize(self,
                 screenshot_path: Optional[str] = None,
                 non_blocking: bool = False):
        """Visualize scene graph using Open3D.
        
        Args:
            screenshot_path: Optional path to save screenshot
            non_blocking: Whether to run the visualizer in non-blocking mode
        """
        # Create Open3D geometries
        geometries = self._create_geometries()
        
        if not geometries:
            logger.warning("No geometries to visualize")
            return
            
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Get render options
        opt = vis.get_render_option()
        opt.background_color = self.background_color
        opt.point_size = self.point_size
        opt.show_coordinate_frame = self.show_coordinate_frame
        
        # Reset view to look at entire scene
        vis.reset_view_point(True)
        
        # Save screenshot if path provided
        if screenshot_path is not None:
            ensure_dir(os.path.dirname(screenshot_path))
            _ = vis.capture_screen_image(screenshot_path, True)
            logger.info(f"Screenshot saved to {screenshot_path}")
        
        # Run visualizer
        if non_blocking:
            vis.run()
        else:
            vis.run()
            vis.destroy_window()
    
    def _create_geometries(self) -> List[Any]:
        """Create Open3D geometries from scene graph.
        
        Returns:
            List of Open3D geometry objects
        """
        geometries = []
        
        # Create point clouds for objects
        for obj_id, obj in self.scene_graph.objects.items():
            # Skip if no points
            if len(obj.points) == 0:
                continue
                
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj.points)
            
            # Set color based on object color
            color = np.array(obj.color) / 255.0  # Convert to [0, 1]
            
            # Highlight if selected
            if obj_id in self.highlight_objects:
                color = self.highlight_color
                
            pcd.paint_uniform_color(color)
            
            # Add to geometries
            geometries.append(pcd)
            
            # Add bounding box if enabled
            if self.show_bounding_boxes:
                bbox = o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=obj.bbox[:3],
                    max_bound=obj.bbox[3:6]
                )
                
                # Set color (slightly transparent version of point cloud color)
                bbox.color = color * 0.8
                
                # Add to geometries
                geometries.append(bbox)
        
        # Add relationship lines if enabled
        if self.show_relationships:
            # Create line set for relationships
            line_points = []
            line_indices = []
            line_colors = []
            
            for rel_id, rel in self.scene_graph.relationships.items():
                # Skip if not in types to show (if filtering is active)
                if self.relationship_types_to_show and rel.type not in self.relationship_types_to_show:
                    continue
                    
                # Get source and target objects
                source_obj = self.scene_graph.get_object(rel.source_id)
                target_obj = self.scene_graph.get_object(rel.target_id)
                
                if source_obj is None or target_obj is None:
                    continue
                
                # Get centers
                source_center = source_obj.center
                target_center = target_obj.center
                
                # Add points
                idx1 = len(line_points)
                idx2 = idx1 + 1
                
                line_points.append(source_center)
                line_points.append(target_center)
                
                # Add line index
                line_indices.append([idx1, idx2])
                
                # Get color for relationship type
                color = self.relationship_colors.get(rel.type, self.relationship_colors['default'])
                line_colors.append(color)
            
            # Create line set if there are relationships
            if line_points:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(line_indices)
                line_set.colors = o3d.utility.Vector3dVector(line_colors)
                
                # Add to geometries
                geometries.append(line_set)
        
        return geometries
    
    def visualize_graph_2d(self, 
                          output_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (16, 12),
                          node_size: int = 700,
                          font_size: int = 10):
        """Visualize scene graph as a 2D network diagram.
        
        Args:
            output_path: Optional path to save the visualization
            figsize: Figure size in inches
            node_size: Size of nodes
            font_size: Size of labels
        """
        # Create a simplified graph for visualization
        G = nx.DiGraph()
        
        # Add nodes
        for obj_id, obj in self.scene_graph.objects.items():
            G.add_node(obj_id, label=obj.label)
        
        # Add edges
        for rel_id, rel in self.scene_graph.relationships.items():
            # Skip if not in types to show (if filtering is active)
            if self.relationship_types_to_show and rel.type not in self.relationship_types_to_show:
                continue
                
            G.add_edge(rel.source_id, rel.target_id, 
                     label=rel.type, 
                     weight=rel.confidence)
        
        # Set up the plot
        plt.figure(figsize=figsize)
        
        # Use spring layout for node positioning
        try:
            pos = nx.spring_layout(G, seed=42, k=0.15, iterations=100)
        except:
            # Fallback if spring layout fails
            pos = nx.circular_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_size=node_size, 
                             node_color='lightblue',
                             alpha=0.8)
        
        # Highlight selected nodes if any
        if self.highlight_objects:
            # Get subset of nodes to highlight
            highlight_nodes = [n for n in G.nodes if n in self.highlight_objects]
            if highlight_nodes:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=highlight_nodes,
                                     node_size=node_size + 100,
                                     node_color='orange',
                                     alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             width=1.5, 
                             alpha=0.5, 
                             arrowsize=15)
        
        # Draw node labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                              labels=node_labels,
                              font_size=font_size,
                              font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(u, v): G.edges[u, v]['label'] 
                       for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, 
                                   edge_labels=edge_labels,
                                   font_size=font_size - 2)
        
        plt.title(f"Scene Graph: {self.scene_graph.scene_info['name']}", fontsize=16)
        plt.axis('off')
        
        # Save or show
        if output_path:
            ensure_dir(os.path.dirname(output_path))
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved graph visualization to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def highlight_path(self, 
                      source_id: int, 
                      target_id: int,
                      output_path: Optional[str] = None):
        """Highlight a path between two objects in the scene graph.
        
        Args:
            source_id: Source object ID
            target_id: Target object ID
            output_path: Optional path to save the visualization
        """
        # Create undirected graph for path finding
        G = nx.Graph()
        
        # Add nodes and edges
        for obj_id in self.scene_graph.objects:
            G.add_node(obj_id)
        
        for rel in self.scene_graph.relationships.values():
            G.add_edge(rel.source_id, rel.target_id, weight=1.0 / rel.confidence)
        
        # Check if both objects exist
        if source_id not in G.nodes or target_id not in G.nodes:
            logger.warning(f"Source or target object not found in graph")
            return
        
        # Find shortest path
        try:
            path = nx.shortest_path(G, source=source_id, target=target_id)
            
            # Set highlight objects
            self.highlight_objects = set(path)
            
            # Set relationship types to show only those on the path
            self.relationship_types_to_show = set()
            
            for i in range(len(path) - 1):
                obj1, obj2 = path[i], path[i+1]
                
                # Find relationships between these objects
                for rel in self.scene_graph.relationships.values():
                    if (rel.source_id == obj1 and rel.target_id == obj2) or \
                       (rel.source_id == obj2 and rel.target_id == obj1):
                        self.relationship_types_to_show.add(rel.type)
            
            # Visualize
            if output_path:
                self.visualize(screenshot_path=output_path)
            else:
                self.visualize()
                
            logger.info(f"Highlighted path from {source_id} to {target_id}")
            
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {source_id} and {target_id}")
    
    def visualize_object_neighborhood(self, 
                                    obj_id: int, 
                                    depth: int = 1,
                                    output_path: Optional[str] = None):
        """Visualize the neighborhood of an object in the scene graph.
        
        Args:
            obj_id: Object ID
            depth: Depth of neighborhood (1 = direct neighbors only)
            output_path: Optional path to save the visualization
        """
        # Check if object exists
        if obj_id not in self.scene_graph.objects:
            logger.warning(f"Object {obj_id} not found in scene graph")
            return
            
        # Create graph
        G = nx.Graph()
        
        # Add all objects as nodes
        for o_id in self.scene_graph.objects:
            G.add_node(o_id)
        
        # Add edges for relationships
        for rel in self.scene_graph.relationships.values():
            G.add_edge(rel.source_id, rel.target_id)
        
        # Get neighborhood
        neighborhood = {obj_id}
        current = {obj_id}
        
        for _ in range(depth):
            next_level = set()
            for node in current:
                next_level.update(G.neighbors(node))
            current = next_level - neighborhood
            neighborhood.update(current)
        
        # Set highlight objects
        self.highlight_objects = {obj_id}  # Just highlight the central object
        
        # Filter objects to show only those in neighborhood
        objects_to_show = self.scene_graph.objects.copy()
        for o_id in list(objects_to_show.keys()):
            if o_id not in neighborhood:
                del objects_to_show[o_id]
        
        # Filter relationships to show only those in neighborhood
        relationships_to_show = self.scene_graph.relationships.copy()
        for r_id, rel in list(relationships_to_show.items()):
            if rel.source_id not in neighborhood or rel.target_id not in neighborhood:
                del relationships_to_show[r_id]
        
        # Create a temporary scene graph with only the neighborhood
        temp_sg = SceneGraph()
        temp_sg.objects = objects_to_show
        temp_sg.relationships = relationships_to_show
        temp_sg.scene_info = self.scene_graph.scene_info.copy()
        temp_sg.scene_info['name'] = f"Neighborhood of {self.scene_graph.objects[obj_id].label}"
        
        # Create a new visualizer for this subgraph
        vis = SceneGraphVisualizer(temp_sg, self.window_name)
        vis.highlight_objects = {obj_id}
        
        # Visualize
        if output_path:
            vis.visualize_graph_2d(output_path=output_path)
        else:
            vis.visualize_graph_2d()
            
        logger.info(f"Visualized neighborhood of object {obj_id} with depth {depth}")


def visualize_scene_graph(scene_graph: SceneGraph,
                        mode: str = "3d",
                        output_path: Optional[str] = None,
                        window_name: str = "Scene Graph Visualization"):
    """Convenience function to visualize a scene graph.
    
    Args:
        scene_graph: Scene graph to visualize
        mode: Visualization mode ('3d' or '2d')
        output_path: Optional path to save visualization
        window_name: Window title
    """
    # Create visualizer
    vis = SceneGraphVisualizer(scene_graph, window_name=window_name)
    
    # Visualize based on mode
    if mode == "2d":
        vis.visualize_graph_2d(output_path=output_path)
    else:
        vis.visualize(screenshot_path=output_path)


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Scene graph visualization")
    parser.add_argument("--scene_graph", required=True, help="Path to scene graph file")
    parser.add_argument("--mode", choices=["3d", "2d"], default="2d", help="Visualization mode")
    parser.add_argument("--output", help="Path to save visualization")
    parser.add_argument("--highlight", type=int, nargs="+", help="IDs of objects to highlight")
    
    args = parser.parse_args()
    
    # Load scene graph
    try:
        scene_graph = SceneGraph.load(args.scene_graph)
        logger.info(f"Loaded scene graph with {len(scene_graph.objects)} objects and {len(scene_graph.relationships)} relationships")
    except Exception as e:
        logger.error(f"Failed to load scene graph: {e}")
        exit(1)
    
    # Create visualizer
    vis = SceneGraphVisualizer(scene_graph)
    
    # Set highlight objects if provided
    if args.highlight:
        vis.highlight_objects = set(args.highlight)
    
    # Visualize
    if args.mode == "2d":
        vis.visualize_graph_2d(output_path=args.output)
    else:
        vis.visualize(screenshot_path=args.output)