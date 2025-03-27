#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mesh visualization utilities for reconstructed 3D meshes.

This module provides tools to visualize triangle meshes, including
semantically labeled meshes with per-vertex or per-face attributes.

Author: Alex Johnson
Date: 2024-01-28
Last modified: 2024-03-05
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from recontext.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

class MeshVisualizer:
    """Visualizer for triangle meshes."""
    
    def __init__(self, 
                 mesh: o3d.geometry.TriangleMesh,
                 window_name: str = "Mesh Visualization"):
        """Initialize mesh visualizer.
        
        Args:
            mesh: Triangle mesh to visualize
            window_name: Window title
        """
        self.mesh = mesh
        self.window_name = window_name
        
        # Default view settings
        self.background_color = [0.1, 0.1, 0.1]  # Dark gray
        self.point_size = 2.0
        self.line_width = 1.0
        self.show_wireframe = False
        self.show_vertices = False
        self.show_coordinate_frame = True
        
        # Default lighting settings
        self.ambient_light = 0.3
        self.directional_light = 0.7
        
        # Additional geometries to visualize
        self.extra_geometries = []
        
    def visualize(self, 
                 screenshot_path: Optional[str] = None,
                 non_blocking: bool = False):
        """Visualize mesh using Open3D.
        
        Args:
            screenshot_path: Optional path to save screenshot
            non_blocking: Whether to run the visualizer in non-blocking mode
        """
        # check if mesh has triangles
        if self.mesh is None or len(self.mesh.triangles) == 0:
            logger.warning("Mesh has no triangles, nothing to visualize")
            return
            
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)
        
        # Add mesh to the visualizer
        vis.add_geometry(self.mesh)
        
        # Add extra geometries
        for geom in self.extra_geometries:
            vis.add_geometry(geom)
        
        # Get render options
        opt = vis.get_render_option()
        opt.background_color = self.background_color
        opt.point_size = self.point_size
        opt.line_width = self.line_width
        opt.show_coordinate_frame = self.show_coordinate_frame
        
        # Set wireframe mode if requested
        if self.show_wireframe:
            opt.mesh_show_wireframe = True
        
        # Set lighting
        opt.light_on = True
        opt.ambient_light = np.array(self.ambient_light)
        
        # Reset view to look at entire geometry
        vis.reset_view_point(True)
        
        # Save screenshot if path provided
        if screenshot_path is not None:
            ensure_dir(screenshot_path.rsplit('/', 1)[0])
            _ = vis.capture_screen_image(screenshot_path, True)
            logger.info(f"Screenshot saved to {screenshot_path}")
        
        # Run visualizer
        if non_blocking:
            vis.run()
        else:
            vis.run()
            vis.destroy_window()
            
    def add_geometry(self, geometry):
        """Add additional geometry to the visualization.
        
        Args:
            geometry: Open3D geometry object
        """
        self.extra_geometries.append(geometry)
            
    def color_by_height(self, 
                       colormap: str = 'viridis',
                       min_height: Optional[float] = None,
                       max_height: Optional[float] = None):
        """Color mesh by height (Y-coordinate).
        
        Args:
            colormap: Matplotlib colormap name
            min_height: Minimum height for colormap
            max_height: Maximum height for colormap
        """
        if not self.mesh.has_vertices():
            logger.warning("Mesh has no vertices")
            return
            
        # Get vertices
        vertices = np.asarray(self.mesh.vertices)
        
        # Extract Y-coordinates (height)
        heights = vertices[:, 1]
        
        # Determine height range
        min_h = min_height if min_height is not None else np.min(heights)
        max_h = max_height if max_height is not None else np.max(heights)
        
        # Normalize heights to [0, 1]
        norm_heights = (heights - min_h) / max(max_h - min_h, 1e-10)
        
        # Get colormap
        cmap = plt.get_cmap(colormap)
        
        # Apply colormap
        colors = cmap(norm_heights)[:, :3]  # RGB only, discard alpha
        
        # Set vertex colors
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Applied height-based coloring with {colormap} colormap")
        
    def color_by_normals(self):
        """Color mesh by surface normals."""
        if not self.mesh.has_vertices():
            logger.warning("Mesh has no vertices")
            return
            
        # Compute normals if not present
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
            
        # Get normals
        normals = np.asarray(self.mesh.vertex_normals)
        
        # Normalize to [0, 1] range
        colors = (normals + 1) / 2.0
        
        # Set vertex colors
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        logger.info("Applied normal-based coloring")
        
    def color_by_labels(self, 
                       vertex_labels: np.ndarray,
                       label_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                       colormap: str = 'tab20'):
        """Color mesh by semantic labels.
        
        Args:
            vertex_labels: Array of vertex labels
            label_colors: Optional dictionary mapping label IDs to RGB colors
            colormap: Matplotlib colormap for automatic color generation
        """
        if not self.mesh.has_vertices():
            logger.warning("Mesh has no vertices")
            return
            
        vertices = np.asarray(self.mesh.vertices)
        
        if len(vertex_labels) != len(vertices):
            logger.error(f"Label count ({len(vertex_labels)}) doesn't match vertex count ({len(vertices)})")
            return
            
        # Get unique labels
        unique_labels = np.unique(vertex_labels)
        
        # Generate colors if not provided
        if label_colors is None:
            # Get colormap
            cmap = plt.get_cmap(colormap)
            
            # Generate colors for each label
            label_colors = {}
            for i, label in enumerate(unique_labels):
                if label >= 0:  # Skip unlabeled (-1)
                    label_colors[label] = cmap(i % cmap.N)[:3]  # RGB only
                else:
                    label_colors[label] = (0.7, 0.7, 0.7)  # Gray for unlabeled
        
        # Apply colors
        colors = np.zeros((len(vertices), 3))
        for label in unique_labels:
            mask = vertex_labels == label
            colors[mask] = label_colors[label]
            
        # Set vertex colors
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Applied label-based coloring with {len(unique_labels)} unique labels")
    
    def create_wireframe(self, color: Tuple[float, float, float] = (0, 0, 0)) -> o3d.geometry.LineSet:
        """Create wireframe from mesh edges.
        
        Args:
            color: Line color
            
        Returns:
            LineSet representing the wireframe
        """
        if not self.mesh.has_triangles():
            logger.warning("Mesh has no triangles")
            return None
            
        # Extract unique edges from triangle faces
        edges = set()
        
        for triangle in np.asarray(self.mesh.triangles):
            # Add edges (smaller index first)
            edges.add((min(triangle[0], triangle[1]), max(triangle[0], triangle[1])))
            edges.add((min(triangle[1], triangle[2]), max(triangle[1], triangle[2])))
            edges.add((min(triangle[2], triangle[0]), max(triangle[2], triangle[0])))
        
        # Convert to numpy arrays
        edge_indices = np.array(list(edges))
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = self.mesh.vertices
        line_set.lines = o3d.utility.Vector2iVector(edge_indices)
        
        # Set color
        colors = np.array([color for _ in range(len(edge_indices))])
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set

    @staticmethod
    def create_from_vertices_and_faces(vertices: np.ndarray, 
                                      faces: np.ndarray, 
                                      vertex_colors: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
        """Create a mesh from vertices and faces.
        
        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
            vertex_colors: Optional Nx3 array of vertex colors
            
        Returns:
            Triangle mesh
        """
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        if vertex_colors is not None:
            if len(vertex_colors) != len(vertices):
                logger.warning(f"Color count ({len(vertex_colors)}) doesn't match vertex count ({len(vertices)})")
            else:
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        # Compute normals
        mesh.compute_vertex_normals()
        
        return mesh


def visualize_mesh(mesh: o3d.geometry.TriangleMesh,
                 vertex_labels: Optional[np.ndarray] = None,
                 label_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                 screenshot_path: Optional[str] = None,
                 window_name: str = "Mesh Visualization"):
    """Convenience function to visualize a mesh.
    
    Args:
        mesh: Triangle mesh to visualize
        vertex_labels: Optional array of vertex labels
        label_colors: Optional dictionary mapping label IDs to RGB colors
        screenshot_path: Optional path to save screenshot
        window_name: Window title
    """
    # Create visualizer
    vis = MeshVisualizer(mesh, window_name=window_name)
    
    # Apply label colors if provided
    if vertex_labels is not None:
        vis.color_by_labels(vertex_labels, label_colors)
    elif not mesh.has_vertex_colors():
        vis.color_by_normals()
    
    # Visualize
    vis.visualize(screenshot_path)


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Mesh visualization")
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--labels", help="Optional vertex labels file")
    parser.add_argument("--color_mode", choices=["labels", "height", "normals"], 
                       default="normals", help="Coloring mode")
    parser.add_argument("--screenshot", help="Path to save screenshot")
    
    args = parser.parse_args()
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    
    # Create visualizer
    vis = MeshVisualizer(mesh)
    
    # Apply coloring
    if args.color_mode == "labels" and args.labels:
        # Load labels
        labels = np.load(args.labels)
        vis.color_by_labels(labels)
    elif args.color_mode == "height":
        vis.color_by_height()
    else:
        vis.color_by_normals()
    
    # Run visualizer
    vis.visualize(args.screenshot)