#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point cloud visualization utilities.

This module provides tools to visualize 3D point clouds with various
coloring schemes and rendering options.

Author: Alex Johnson
Date: 2024-01-20
Last modified: 2024-03-02
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Union, Any

from recontext.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

class PointCloudVisualizer:
    """Visualizer for 3D point clouds."""
    
    def __init__(self, 
                 pointcloud: o3d.geometry.PointCloud,
                 window_name: str = "Point Cloud Visualization"):
        """Initialize point cloud visualizer.
        
        Args:
            pointcloud: Point cloud to visualize
            window_name: Window title
        """
        self.pointcloud = pointcloud
        self.window_name = window_name
        
        # Default view settings
        self.background_color = [0.1, 0.1, 0.1]  # Dark gray
        self.point_size = 2.0
        self.show_coordinate_frame = True
        
        # Additional geometries to visualize
        self.extra_geometries = []
        
    def visualize(self, 
                 screenshot_path: Optional[str] = None,
                 non_blocking: bool = False):
        """Visualize point cloud using Open3D.
        
        Args:
            screenshot_path: Optional path to save screenshot
            non_blocking: Whether to run the visualizer in non-blocking mode
        """
        # Check if point cloud has points
        if self.pointcloud is None or len(self.pointcloud.points) == 0:
            logger.warning("Point cloud has no points, nothing to visualize")
            return
            
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=self.window_name)
        
        # Add point cloud to the visualizer
        vis.add_geometry(self.pointcloud)
        
        # Add extra geometries
        for geom in self.extra_geometries:
            vis.add_geometry(geom)
        
        # Get render options
        opt = vis.get_render_option()
        opt.background_color = self.background_color
        opt.point_size = self.point_size
        opt.show_coordinate_frame = self.show_coordinate_frame
        
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
        """Color point cloud by height (Y-coordinate).
        
        Args:
            colormap: Matplotlib colormap name
            min_height: Minimum height for colormap
            max_height: Maximum height for colormap
        """
        if not self.pointcloud.has_points():
            logger.warning("Point cloud has no points")
            return
            
        # Get points
        points = np.asarray(self.pointcloud.points)
        
        # Extract Y-coordinates (height)
        heights = points[:, 1]
        
        # Determine height range
        min_h = min_height if min_height is not None else np.min(heights)
        max_h = max_height if max_height is not None else np.max(heights)
        
        # Normalize heights to [0, 1]
        norm_heights = np.clip((heights - min_h) / max(max_h - min_h, 1e-10), 0, 1)
        
        # Get colormap
        cmap = plt.get_cmap(colormap)
        
        # Apply colormap
        colors = cmap(norm_heights)[:, :3]  # RGB only, discard alpha
        
        # Set point colors
        self.pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Applied height-based coloring with {colormap} colormap")
    
    def color_by_normals(self):
        """Color point cloud by surface normals."""
        if not self.pointcloud.has_points():
            logger.warning("Point cloud has no points")
            return
            
        # Compute normals if not present
        if not self.pointcloud.has_normals():
            self.pointcloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
        # Get normals
        normals = np.asarray(self.pointcloud.normals)
        
        # Normalize to [0, 1] range
        colors = (normals + 1) / 2.0
        
        # Set point colors
        self.pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info("Applied normal-based coloring")
    
    def color_by_labels(self, 
                       point_labels: np.ndarray,
                       label_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                       colormap: str = 'tab20'):
        """Color point cloud by semantic labels.
        
        Args:
            point_labels: Array of point labels
            label_colors: Optional dictionary mapping label IDs to RGB colors
            colormap: Matplotlib colormap for automatic color generation
        """
        if not self.pointcloud.has_points():
            logger.warning("Point cloud has no points")
            return
            
        points = np.asarray(self.pointcloud.points)
        
        if len(point_labels) != len(points):
            logger.error(f"Label count ({len(point_labels)}) doesn't match point count ({len(points)})")
            return
            
        # Get unique labels
        unique_labels = np.unique(point_labels)
        
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
        colors = np.zeros((len(points), 3))
        for label in unique_labels:
            mask = point_labels == label
            colors[mask] = label_colors[label]
            
        # Set point colors
        self.pointcloud.colors = o3d.utility.Vector3dVector(colors)
        
        logger.info(f"Applied label-based coloring with {len(unique_labels)} unique labels")
    
    def add_labels(self, 
                  labels: List[str], 
                  positions: np.ndarray, 
                  colors: Optional[List[Tuple[float, float, float]]] = None):
        """Add text labels to the visualization.
        
        Note: Open3D doesn't support text rendering directly, so this creates small
        spheres as placeholders. In a proper viewer implementation, these would be
        replaced with actual text.
        
        Args:
            labels: List of text labels
            positions: Nx3 array of label positions
            colors: Optional list of label colors
        """
        if len(labels) != len(positions):
            logger.error(f"Label count ({len(labels)}) doesn't match position count ({len(positions)})")
            return
            
        # Use default colors if not provided
        if colors is None:
            colors = [(1, 0, 0) for _ in labels]  # Default red
            
        # Create small spheres for each label
        for i, (label, pos) in enumerate(zip(labels, positions)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(pos)
            sphere.paint_uniform_color(colors[i % len(colors)])
            
            self.add_geometry(sphere)
    
    def add_bounding_boxes(self, 
                         boxes: List[np.ndarray], 
                         colors: Optional[List[Tuple[float, float, float]]] = None):
        """Add 3D bounding boxes to the visualization.
        
        Args:
            boxes: List of bounding boxes [min_x, min_y, min_z, max_x, max_y, max_z]
            colors: Optional list of box colors
        """
        # Use default colors if not provided
        if colors is None:
            colors = [(0, 1, 0) for _ in boxes]  # Default green
            
        # Create bounding boxes
        for i, box in enumerate(boxes):
            min_bound = box[:3]
            max_bound = box[3:6]
            
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            bbox.color = colors[i % len(colors)]
            
            self.add_geometry(bbox)
    
    def add_graph_edges(self, 
                       start_points: np.ndarray, 
                       end_points: np.ndarray, 
                       colors: Optional[List[Tuple[float, float, float]]] = None):
        """Add graph edges to the visualization.
        
        Args:
            start_points: Nx3 array of edge start points
            end_points: Nx3 array of edge end points
            colors: Optional list of edge colors
        """
        if len(start_points) != len(end_points):
            logger.error(f"Start point count ({len(start_points)}) doesn't match end point count ({len(end_points)})")
            return
            
        # Use default colors if not provided
        if colors is None:
            colors = [(0, 0, 1) for _ in range(len(start_points))]  # Default blue
            
        # Create line set
        line_set = o3d.geometry.LineSet()
        
        # Set points (all unique start and end points)
        all_points = np.vstack([start_points, end_points])
        unique_points, inverse = np.unique(all_points, axis=0, return_inverse=True)
        
        line_set.points = o3d.utility.Vector3dVector(unique_points)
        
        # Set lines
        lines = []
        for i in range(len(start_points)):
            # Get indices in unique_points
            start_idx = inverse[i]
            end_idx = inverse[i + len(start_points)]
            lines.append([start_idx, end_idx])
            
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Set colors
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        self.add_geometry(line_set)
        
    @staticmethod
    def create_from_points(points: np.ndarray, 
                         colors: Optional[np.ndarray] = None,
                         normals: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
        """Create a point cloud from points.
        
        Args:
            points: Nx3 array of point coordinates
            colors: Optional Nx3 array of point colors (RGB, range [0, 1])
            normals: Optional Nx3 array of point normals
            
        Returns:
            Point cloud
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Set colors if provided
        if colors is not None:
            if len(colors) != len(points):
                logger.warning(f"Color count ({len(colors)}) doesn't match point count ({len(points)})")
            else:
                pcd.colors = o3d.utility.Vector3dVector(colors)
                
        # Set normals if provided
        if normals is not None:
            if len(normals) != len(points):
                logger.warning(f"Normal count ({len(normals)}) doesn't match point count ({len(points)})")
            else:
                pcd.normals = o3d.utility.Vector3dVector(normals)
                
        return pcd


def visualize_pointcloud(pointcloud: o3d.geometry.PointCloud,
                       point_labels: Optional[np.ndarray] = None,
                       label_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
                       screenshot_path: Optional[str] = None,
                       window_name: str = "Point Cloud Visualization"):
    """Convenience function to visualize a point cloud.
    
    Args:
        pointcloud: Point cloud to visualize
        point_labels: Optional array of point labels
        label_colors: Optional dictionary mapping label IDs to RGB colors
        screenshot_path: Optional path to save screenshot
        window_name: Window title
    """
    # Create visualizer
    vis = PointCloudVisualizer(pointcloud, window_name=window_name)
    
    # Apply label colors if provided
    if point_labels is not None:
        vis.color_by_labels(point_labels, label_colors)
    elif not pointcloud.has_colors():
        vis.color_by_height()
    
    # Visualize
    vis.visualize(screenshot_path)


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Point cloud visualization")
    parser.add_argument("--pointcloud", required=True, help="Input point cloud file")
    parser.add_argument("--labels", help="Optional point labels file")
    parser.add_argument("--color_mode", choices=["labels", "height", "normals"], 
                       default="height", help="Coloring mode")
    parser.add_argument("--screenshot", help="Path to save screenshot")
    
    args = parser.parse_args()
    
    # Load point cloud
    try:
        pcd = o3d.io.read_point_cloud(args.pointcloud)
        logger.info(f"Loaded point cloud with {len(pcd.points)} points")
    except Exception as e:
        logger.error(f"Failed to load point cloud: {e}")
        exit(1)
    
    # Create visualizer
    vis = PointCloudVisualizer(pcd)
    
    # Apply coloring
    if args.color_mode == "labels" and args.labels:
        # Load labels
        try:
            labels = np.load(args.labels)
            vis.color_by_labels(labels)
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            vis.color_by_height()  # Fallback
    elif args.color_mode == "normals":
        vis.color_by_normals()
    else:
        vis.color_by_height()
    
    # Run visualizer
    vis.visualize(args.screenshot)