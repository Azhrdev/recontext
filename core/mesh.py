#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mesh generation and processing module.

This module provides functions for creating and improving triangle meshes from 
3D point cloud data, including surface reconstruction, mesh filtering, and 
texture mapping.

Author: Alex Johnson
Date: 2024-01-25
Last modified: 2024-03-10
"""

import numpy as np
import open3d as o3d
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from recontext.core.pointcloud import compute_normals
from recontext.utils.transforms import normalize_pointcloud

logger = logging.getLogger(__name__)

def create_mesh(pointcloud: o3d.geometry.PointCloud,
               method: str = "poisson",
               depth: int = 9,
               scale: float = 1.1,
               alpha: float = 0.03) -> o3d.geometry.TriangleMesh:
    """Create triangle mesh from point cloud.
    
    Args:
        pointcloud: Input point cloud
        method: Reconstruction method ('poisson', 'alpha_shape', or 'ball_pivot')
        depth: Depth parameter for Poisson reconstruction
        scale: Scale parameter for Poisson reconstruction
        alpha: Alpha parameter for Alpha Shape reconstruction
    
    Returns:
        Triangle mesh
    """
    # Ensure point cloud has normals
    if not pointcloud.has_normals():
        logger.info("Computing point cloud normals")
        pointcloud = compute_normals(pointcloud)
    
    # Create mesh based on selected method
    if method == "poisson":
        logger.info(f"Creating mesh using Poisson reconstruction (depth={depth})")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pointcloud, depth=depth, scale=scale, linear_fit=False)
        
        # Filter low-density vertices
        try:
            vertices_to_remove = densities < np.quantile(densities, 0.01)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        except Exception as e:
            logger.warning(f"Failed to filter low-density vertices: {e}")
    
    elif method == "alpha_shape":
        logger.info(f"Creating mesh using Alpha Shape (alpha={alpha})")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pointcloud, alpha=alpha)
    
    elif method == "ball_pivot":
        logger.info(f"Creating mesh using Ball Pivoting")
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pointcloud, o3d.utility.DoubleVector(radii))
    
    else:
        logger.error(f"Unknown reconstruction method: {method}")
        # Fall back to Alpha Shape
        logger.warning("Falling back to Alpha Shape reconstruction")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pointcloud, alpha=alpha)
    
    # Transfer colors from pointcloud to mesh if available
    if pointcloud.has_colors() and len(mesh.vertices) > 0:
        try:
            # This is approximate and might not always work perfectly
            # A better approach would be to use the vertex-to-point mapping from the reconstruction
            pcd_tree = o3d.geometry.KDTreeFlann(pointcloud)
            vertex_colors = []
            
            for vertex in tqdm(np.asarray(mesh.vertices), desc="Transferring colors"):
                _, idx, _ = pcd_tree.search_knn_vector_3d(vertex, 1)
                if idx:
                    color = np.asarray(pointcloud.colors)[idx[0]]
                    vertex_colors.append(color)
                else:
                    vertex_colors.append([0.7, 0.7, 0.7])  # Default gray
            
            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        except Exception as e:
            logger.warning(f"Failed to transfer colors: {e}")
    
    logger.info(f"Created mesh with {len(mesh.triangles)} triangles")
    
    return mesh


def improve_mesh(mesh: o3d.geometry.TriangleMesh,
                fill_holes: bool = True,
                max_hole_size: int = 100,
                remove_outliers: bool = True,
                remove_duplicates: bool = True,
                smooth: bool = True,
                smooth_iterations: int = 5) -> o3d.geometry.TriangleMesh:
    """Improve mesh quality by filling holes and removing artifacts.
    
    Args:
        mesh: Input triangle mesh
        fill_holes: Whether to fill holes
        max_hole_size: Maximum hole size to fill (in edges)
        remove_outliers: Whether to remove outlier triangles
        remove_duplicates: Whether to remove duplicate triangles and vertices
        smooth: Whether to apply Laplacian smoothing
        smooth_iterations: Number of smoothing iterations
    
    Returns:
        Improved triangle mesh
    """
    # Create a copy of the input mesh
    result = o3d.geometry.TriangleMesh(mesh)
    
    # Track mesh statistics
    initial_triangles = len(result.triangles)
    initial_vertices = len(result.vertices)
    
    # Remove duplicate vertices and triangles
    if remove_duplicates:
        result.remove_duplicated_vertices()
        result.remove_duplicated_triangles()
        logger.info("Removed duplicate vertices and triangles")
    
    # Ensure mesh is manifold for operations
    try:
        result.compute_vertex_normals()
        result.compute_triangle_normals()
    except Exception as e:
        logger.warning(f"Failed to compute normals: {e}")
    
    # Remove outlier triangles
    if remove_outliers:
        # Identify outliers based on distance to neighboring triangles
        # This is a simplified approach; more sophisticated methods exist
        try:
            # Compute triangle centroids
            triangle_indices = np.asarray(result.triangles)
            vertices = np.asarray(result.vertices)
            centroids = np.mean([vertices[triangle_indices[:, 0]],
                                vertices[triangle_indices[:, 1]],
                                vertices[triangle_indices[:, 2]]], axis=0)
            
            # Create point cloud from centroids
            centroid_cloud = o3d.geometry.PointCloud()
            centroid_cloud.points = o3d.utility.Vector3dVector(centroids)
            
            # Use statistical outlier removal
            _, inlier_indices = centroid_cloud.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)
            
            if inlier_indices:
                # Create new mesh with only inlier triangles
                inlier_triangles = np.asarray(result.triangles)[inlier_indices]
                result.triangles = o3d.utility.Vector3iVector(inlier_triangles)
                # Clean up unused vertices
                result.remove_unreferenced_vertices()
                logger.info(f"Removed {initial_triangles - len(result.triangles)} outlier triangles")
        except Exception as e:
            logger.warning(f"Failed to remove outlier triangles: {e}")
    
    # Fill holes
    if fill_holes:
        try:
            holes = result.get_non_manifold_edges()
            if len(holes) > 0:
                result.fill_holes(max_hole_size)
                new_holes = result.get_non_manifold_edges()
                logger.info(f"Filled {len(holes) - len(new_holes)} holes")
        except Exception as e:
            logger.warning(f"Failed to fill holes: {e}")
    
    # Apply smoothing
    if smooth and smooth_iterations > 0:
        try:
            for _ in range(smooth_iterations):
                result = result.filter_smooth_laplacian(1)
            logger.info(f"Applied {smooth_iterations} iterations of Laplacian smoothing")
        except Exception as e:
            logger.warning(f"Failed to apply smoothing: {e}")
    
    # Recompute normals
    try:
        result.compute_vertex_normals()
        result.compute_triangle_normals()
    except Exception as e:
        logger.warning(f"Failed to compute normals: {e}")
    
    # Log mesh statistics
    final_triangles = len(result.triangles)
    final_vertices = len(result.vertices)
    logger.info(f"Mesh improvement: {initial_triangles} → {final_triangles} triangles, "
               f"{initial_vertices} → {final_vertices} vertices")
    
    return result


def simplify_mesh(mesh: o3d.geometry.TriangleMesh,
                 target_triangles: int = 100000,
                 preserve_details: bool = True) -> o3d.geometry.TriangleMesh:
    """Simplify mesh by reducing the number of triangles.
    
    Args:
        mesh: Input triangle mesh
        target_triangles: Target number of triangles
        preserve_details: Whether to preserve mesh details
    
    Returns:
        Simplified triangle mesh
    """
    # Create a copy of the input mesh
    result = o3d.geometry.TriangleMesh(mesh)
    
    # Get initial triangle count
    initial_triangles = len(result.triangles)
    
    # Calculate reduction ratio
    if target_triangles >= initial_triangles:
        logger.info(f"Target triangle count {target_triangles} is greater than current count {initial_triangles}")
        return result
    
    reduction_ratio = target_triangles / initial_triangles
    
    # Apply mesh simplification
    result = result.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    
    # Log results
    final_triangles = len(result.triangles)
    logger.info(f"Mesh simplification: {initial_triangles} → {final_triangles} triangles "
               f"(target: {target_triangles}, ratio: {reduction_ratio:.3f})")
    
    # Recompute normals
    try:
        result.compute_vertex_normals()
        result.compute_triangle_normals()
    except Exception as e:
        logger.warning(f"Failed to compute normals: {e}")
    
    return result


def texture_map_mesh(mesh: o3d.geometry.TriangleMesh,
                    images: List[np.ndarray],
                    cameras: Dict[int, Any],
                    texture_size: Tuple[int, int] = (2048, 2048)) -> o3d.geometry.TriangleMesh:
    """Map textures from images onto mesh.
    
    This is a placeholder function. Proper texture mapping is complex and requires
    UV unwrapping, projection of images, blending, etc.
    
    Args:
        mesh: Input triangle mesh
        images: List of input images
        cameras: Dictionary of cameras by image ID
        texture_size: Size of the texture image
    
    Returns:
        Textured triangle mesh
    """
    logger.warning("Full texture mapping not implemented")
    
    # This is a stub implementation that creates vertex colors instead of a proper texture
    logger.info("Creating vertex colors as fallback for texture mapping")
    
    # Create a copy of the input mesh
    result = o3d.geometry.TriangleMesh(mesh)
    
    # Get vertices
    vertices = np.asarray(result.vertices)
    
    # Initialize vertex colors
    vertex_colors = np.zeros((len(vertices), 3))
    weights = np.zeros(len(vertices))
    
    # Project vertices to each camera view
    for image_id, camera in cameras.items():
        # Skip if no image data
        if image_id >= len(images) or images[image_id] is None:
            continue
        
        image = images[image_id]
        height, width = image.shape[:2]
        
        try:
            # Project vertices to image
            vertices_2d, depths = camera.project(vertices)
            
            # For each vertex
            for i, (x, y) in enumerate(vertices_2d):
                # Check if visible in this view
                if 0 <= x < width and 0 <= y < height and depths[i] > 0:
                    # Get color from image
                    if len(image.shape) > 2:  # Color image
                        color = image[int(y), int(x)]
                    else:  # Grayscale image
                        gray = image[int(y), int(x)]
                        color = np.array([gray, gray, gray])
                    
                    # Weight by inverse depth
                    weight = 1.0 / max(depths[i], 1e-6)
                    
                    # Update weighted color
                    vertex_colors[i] += color * weight
                    weights[i] += weight
        except Exception as e:
            logger.warning(f"Error projecting to camera {image_id}: {e}")
    
    # Normalize colors
    mask = weights > 0
    vertex_colors[mask] = vertex_colors[mask] / weights[mask].reshape(-1, 1)
    
    # Set default color for points without weights
    vertex_colors[~mask] = [200, 200, 200]  # Default gray
    
    # Set vertex colors
    result.vertex_colors = o3d.utility.Vector3dVector(vertex_colors / 255.0)
    
    logger.info(f"Created vertex colors for {np.sum(mask)} out of {len(vertices)} vertices")
    
    return result


def extract_mesh_features(mesh: o3d.geometry.TriangleMesh) -> Dict[str, Any]:
    """Extract geometric features from mesh.
    
    Args:
        mesh: Input triangle mesh
    
    Returns:
        Dictionary of mesh features
    """
    # Initialize features dictionary
    features = {}
    
    # Basic counts
    features['vertex_count'] = len(mesh.vertices)
    features['triangle_count'] = len(mesh.triangles)
    
    # Compute bounding box
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    features['bounding_box'] = {
        'min_x': float(min_bound[0]),
        'min_y': float(min_bound[1]),
        'min_z': float(min_bound[2]),
        'max_x': float(max_bound[0]),
        'max_y': float(max_bound[1]),
        'max_z': float(max_bound[2])
    }
    
    # Compute dimensions
    dimensions = max_bound - min_bound
    features['dimensions'] = {
        'width': float(dimensions[0]),
        'height': float(dimensions[1]),
        'depth': float(dimensions[2])
    }
    
    # Compute center
    center = mesh.get_center()
    features['center'] = {
        'x': float(center[0]),
        'y': float(center[1]),
        'z': float(center[2])
    }
    
    # Compute volume (if watertight)
    try:
        if mesh.is_watertight():
            features['volume'] = float(mesh.get_volume())
        else:
            features['volume'] = None
    except Exception:
        features['volume'] = None
    
    # Compute surface area
    try:
        features['surface_area'] = float(mesh.get_surface_area())
    except Exception:
        features['surface_area'] = None
    
    # Check manifold status
    features['manifold'] = {
        'is_watertight': mesh.is_watertight(),
        'is_edge_manifold': mesh.is_edge_manifold(),
        'self_intersecting': mesh.is_self_intersecting()
    }
    
    return features


def create_mesh_from_depth_maps(depth_maps: List[np.ndarray],
                               cameras: Dict[int, Any],
                               voxel_size: float = 0.01,
                               sdf_trunc: float = 0.04,
                               depth_scale: float = 1000.0,
                               depth_max: float = 3.0) -> o3d.geometry.TriangleMesh:
    """Create mesh using TSDF integration from depth maps.
    
    Args:
        depth_maps: List of depth maps
        cameras: Dictionary of cameras by image ID
        voxel_size: Size of voxel in TSDF volume
        sdf_trunc: Truncation distance for signed distance function
        depth_scale: Scaling factor for depth values
        depth_max: Maximum depth value
    
    Returns:
        Triangle mesh
    """
    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Integrate depth maps
    for i, depth_map in enumerate(depth_maps):
        # Skip if no depth map or camera
        if depth_map is None or i not in cameras:
            continue
        
        # Get camera
        camera = cameras[i]
        
        # Convert depth map to o3d format
        depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
        
        # Create RGB image (black if not available)
        height, width = depth_map.shape
        color_o3d = o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8))
        
        # Create intrinsic parameter
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height,
            camera.intrinsics.fx, camera.intrinsics.fy,
            camera.intrinsics.cx, camera.intrinsics.cy
        )
        
        # Create extrinsic parameter (4x4 transformation matrix)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = camera.extrinsics.R
        extrinsic[:3, 3] = camera.extrinsics.t.flatten()
        
        # Integrate into volume
        volume.integrate(
            depth_o3d,
            color_o3d,
            intrinsic,
            extrinsic
        )
    
    # Extract mesh
    mesh = volume.extract_triangle_mesh()
    
    # Ensure mesh has normals
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    logger.info(f"Created mesh with {len(mesh.triangles)} triangles from TSDF integration")
    
    return mesh


def segment_mesh(mesh: o3d.geometry.TriangleMesh,
               min_segment_size: int = 100) -> List[o3d.geometry.TriangleMesh]:
    """Segment mesh into connected components.
    
    Args:
        mesh: Input triangle mesh
        min_segment_size: Minimum number of triangles in a segment
    
    Returns:
        List of mesh segments
    """
    # Create a copy of the input mesh
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    
    # Cluster connected triangles
    try:
        triangle_clusters, cluster_n_triangles, _ = mesh_copy.cluster_connected_triangles()
        triangle_clusters = np.array(triangle_clusters)
        cluster_n_triangles = np.array(cluster_n_triangles)
        
        # Keep only clusters with enough triangles
        valid_clusters = cluster_n_triangles >= min_segment_size
        valid_cluster_ids = np.where(valid_clusters)[0]
        
        # Create a mesh for each valid cluster
        segments = []
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        for cluster_id in valid_cluster_ids:
            # Get triangles in this cluster
            cluster_triangles = triangles[triangle_clusters == cluster_id]
            
            # Create new mesh
            segment = o3d.geometry.TriangleMesh()
            segment.vertices = o3d.utility.Vector3dVector(vertices)
            segment.triangles = o3d.utility.Vector3iVector(cluster_triangles)
            
            # Remove unreferenced vertices
            segment.remove_unreferenced_vertices()
            
            # Copy vertex colors if available
            if mesh.has_vertex_colors():
                # This is approximate since we've removed unreferenced vertices
                # A better approach would track the mapping
                if segment.has_vertex_colors():
                    segment.vertex_colors = o3d.utility.Vector3dVector(
                        np.ones((len(segment.vertices), 3)) * 0.7)  # Default gray
            
            segments.append(segment)
        
        logger.info(f"Segmented mesh into {len(segments)} components")
        return segments
        
    except Exception as e:
        logger.error(f"Failed to segment mesh: {e}")
        return [mesh_copy]  # Return copy of original mesh as fallback


def export_mesh_to_file(mesh: o3d.geometry.TriangleMesh,
                       filepath: str,
                       write_vertex_normals: bool = True,
                       write_vertex_colors: bool = True,
                       write_triangle_uvs: bool = False) -> bool:
    """Export mesh to file.
    
    Args:
        mesh: Triangle mesh to export
        filepath: Output file path
        write_vertex_normals: Whether to write vertex normals
        write_vertex_colors: Whether to write vertex colors
        write_triangle_uvs: Whether to write triangle UVs
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Determine file format
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Write mesh
        success = o3d.io.write_triangle_mesh(
            filepath,
            mesh,
            write_vertex_normals=write_vertex_normals,
            write_vertex_colors=write_vertex_colors,
            write_triangle_uvs=write_triangle_uvs
        )
        
        if success:
            logger.info(f"Mesh exported to {filepath}")
        else:
            logger.error(f"Failed to export mesh to {filepath}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error exporting mesh: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Mesh processing")
    parser.add_argument("--input", required=True, help="Input point cloud or mesh file")
    parser.add_argument("--output", required=True, help="Output mesh file")
    parser.add_argument("--method", choices=['poisson', 'alpha_shape', 'ball_pivot'],
                       default='poisson', help="Mesh creation method")
    parser.add_argument("--improve", action="store_true", help="Apply mesh improvement")
    parser.add_argument("--simplify", type=int, help="Target triangle count for simplification")
    
    args = parser.parse_args()
    
    # Determine input type based on extension
    import os
    file_ext = os.path.splitext(args.input)[1].lower()
    
    # Load input
    if file_ext in ['.ply', '.pcd', '.xyz']:
        # Load as point cloud
        pcd = o3d.io.read_point_cloud(args.input)
        logger.info(f"Loaded point cloud with {len(pcd.points)} points")
        
        # Create mesh
        mesh = create_mesh(pcd, method=args.method)
        
    elif file_ext in ['.obj', '.off', '.stl']:
        # Load as mesh
        mesh = o3d.io.read_triangle_mesh(args.input)
        logger.info(f"Loaded mesh with {len(mesh.triangles)} triangles")
        
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        import sys
        sys.exit(1)
    
    # Apply operations
    if args.improve:
        mesh = improve_mesh(mesh)
    
    if args.simplify:
        mesh = simplify_mesh(mesh, target_triangles=args.simplify)
    
    # Save result
    export_mesh_to_file(mesh, args.output)