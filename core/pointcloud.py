#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Point cloud processing module for filtering, densification and analysis.

Author: James Wei
Date: 2024-01-18
Last modified: 2024-03-15
"""

import numpy as np
import open3d as o3d
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from recontext.core.camera import Camera
from recontext.utils.transforms import normalize_pointcloud

logger = logging.getLogger(__name__)

def filter_pointcloud(pointcloud: o3d.geometry.PointCloud,
                     min_neighbors: int = 5,
                     radius: float = 0.05,
                     statistical_outlier: bool = True,
                     nb_neighbors: int = 20,
                     std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """Filter outliers from point cloud.
    
    Args:
        pointcloud: Input point cloud
        min_neighbors: Minimum number of neighbors for radius filter
        radius: Radius for searching neighbors
        statistical_outlier: Whether to apply statistical outlier removal
        nb_neighbors: Number of neighbors for statistical outlier removal
        std_ratio: Standard deviation ratio for statistical outlier removal
        
    Returns:
        Filtered point cloud
    """
    # Create a copy of the input point cloud
    filtered = o3d.geometry.PointCloud(pointcloud)
    
    # Apply radius outlier removal
    filtered, _ = filtered.remove_radius_outlier(
        nb_points=min_neighbors, radius=radius)
    
    # Apply statistical outlier removal if requested
    if statistical_outlier:
        filtered, _ = filtered.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    logger.info(f"Filtered point cloud from {len(pointcloud.points)} to {len(filtered.points)} points")
    
    return filtered


def segment_pointcloud(pointcloud: o3d.geometry.PointCloud,
                      eps: float = 0.05,
                      min_points: int = 10) -> List[np.ndarray]:
    """Segment point cloud into clusters using DBSCAN.
    
    Args:
        pointcloud: Input point cloud
        eps: Maximum distance between points in a cluster
        min_points: Minimum number of points in a cluster
        
    Returns:
        List of point indices for each cluster
    """
    # Get points as numpy array
    points = np.asarray(pointcloud.points)
    
    # Apply DBSCAN clustering
    db = o3d.geometry.PointCloud.cluster_dbscan(
        pointcloud, eps=eps, min_points=min_points, print_progress=False)
    
    # Convert labels to numpy array
    labels = np.array(db)
    
    # Get unique labels (excluding noise label -1)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]
    
    # Create list of point indices for each cluster
    clusters = [np.where(labels == label)[0] for label in unique_labels]
    
    logger.info(f"Segmented point cloud into {len(clusters)} clusters")
    
    return clusters


def compute_normals(pointcloud: o3d.geometry.PointCloud,
                   radius: float = 0.1,
                   max_nn: int = 30,
                   orient_consistent: bool = True) -> o3d.geometry.PointCloud:
    """Compute normals for point cloud.
    
    Args:
        pointcloud: Input point cloud
        radius: Radius for searching neighbors
        max_nn: Maximum number of neighbors
        orient_consistent: Whether to orient normals consistently
        
    Returns:
        Point cloud with normals
    """
    # Create a copy of the input point cloud
    result = o3d.geometry.PointCloud(pointcloud)
    
    # Compute normals
    result.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    
    # Orient normals consistently if requested
    if orient_consistent:
        result.orient_normals_consistent_tangent_plane(k=max_nn)
    
    logger.info(f"Computed normals for {len(result.points)} points")
    
    return result


def compute_fpfh_features(pointcloud: o3d.geometry.PointCloud,
                         radius: float = 0.1) -> np.ndarray:
    """Compute FPFH features for point cloud.
    
    Args:
        pointcloud: Input point cloud with normals
        radius: Radius for searching neighbors
        
    Returns:
        FPFH features for each point
    """
    # Check if normals are available
    if not pointcloud.has_normals():
        logger.warning("Point cloud does not have normals, computing now")
        pointcloud = compute_normals(pointcloud)
    
    # Compute FPFH features
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pointcloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    
    # Convert to numpy array
    features = np.asarray(fpfh.data).T
    
    logger.info(f"Computed FPFH features of shape {features.shape}")
    
    return features


def compute_depth_map(pointcloud: o3d.geometry.PointCloud,
                     camera: Camera,
                     width: int,
                     height: int) -> np.ndarray:
    """Project point cloud to camera and create depth map.
    
    Args:
        pointcloud: Input point cloud
        camera: Camera for projection
        width: Width of depth map
        height: Height of depth map
        
    Returns:
        Depth map
    """
    # Get points as numpy array
    points = np.asarray(pointcloud.points)
    
    # Project points to camera
    points_2d, depths = camera.project(points)
    
    # Initialize depth map with zeros
    depth_map = np.zeros((height, width), dtype=np.float32)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    
    # Iterate through projected points
    for i, (x, y) in enumerate(points_2d):
        # Check if point is inside image
        if 0 <= x < width and 0 <= y < height:
            # Round to integer coordinates
            x_int, y_int = int(x), int(y)
            
            # Update depth map if this point is closer
            if depths[i] < depth_buffer[y_int, x_int]:
                depth_map[y_int, x_int] = depths[i]
                depth_buffer[y_int, x_int] = depths[i]
    
    # Replace infinite values with zeros
    depth_map[depth_buffer == np.inf] = 0
    
    return depth_map


def pointcloud_from_depth(depth_map: np.ndarray,
                         camera: Camera,
                         mask: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """Create point cloud from depth map.
    
    Args:
        depth_map: Input depth map
        camera: Camera for back-projection
        mask: Optional mask for valid depth values
        
    Returns:
        Point cloud
    """
    # Get image dimensions
    height, width = depth_map.shape
    
    # Create grid of pixel coordinates
    y, x = np.mgrid[0:height, 0:width]
    points_2d = np.column_stack((x.ravel(), y.ravel()))
    depths = depth_map.ravel()
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.ravel()
        valid = np.logical_and(depths > 0, mask_flat > 0)
    else:
        valid = depths > 0
    
    points_2d = points_2d[valid]
    depths = depths[valid]
    
    # Back-project points
    points_3d = camera.backproject(points_2d, depths)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    logger.info(f"Created point cloud with {len(pcd.points)} points from depth map")
    
    return pcd


def densify_pointcloud(images: List[np.ndarray],
                      cameras: Dict[int, Camera],
                      sparse_pointcloud: o3d.geometry.PointCloud,
                      patch_size: int = 11,
                      num_consistent_views: int = 3,
                      depth_threshold: float = 0.01) -> o3d.geometry.PointCloud:
    """Densify sparse point cloud using patch-match stereo.
    
    This is a simplified version of MVS (Multi-View Stereo) that
    creates a denser point cloud from multiple calibrated images.
    
    Args:
        images: List of input images
        cameras: Dictionary of cameras by image ID
        sparse_pointcloud: Sparse point cloud from SfM
        patch_size: Size of patch for matching
        num_consistent_views: Minimum number of consistent views
        depth_threshold: Threshold for depth consistency
        
    Returns:
        Dense point cloud
    """
    logger.info("Starting point cloud densification")
    
    # Create result point cloud
    dense_cloud = o3d.geometry.PointCloud()
    
    # Process each image as reference
    for ref_id, ref_camera in tqdm(cameras.items(), desc="Densifying point cloud"):
        # Skip if no image data
        if ref_id >= len(images):
            continue
        
        ref_image = images[ref_id]
        if ref_image is None:
            continue
        
        # Convert to grayscale if needed
        if len(ref_image.shape) > 2:
            ref_image_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        else:
            ref_image_gray = ref_image
        
        height, width = ref_image_gray.shape
        
        # Select source views
        source_views = []
        for src_id, src_camera in cameras.items():
            if src_id == ref_id:
                continue
            
            # Skip if no image data
            if src_id >= len(images):
                continue
            
            src_image = images[src_id]
            if src_image is None:
                continue
            
            # Compute baseline
            ref_center = -ref_camera.extrinsics.R.T @ ref_camera.extrinsics.t
            src_center = -src_camera.extrinsics.R.T @ src_camera.extrinsics.t
            baseline = np.linalg.norm(src_center - ref_center)
            
            # Add to source views if baseline is non-zero
            if baseline > 1e-6:
                source_views.append((src_id, baseline))
        
        # Skip if not enough source views
        if len(source_views) < num_consistent_views:
            continue
        
        # Sort source views by baseline (descending)
        source_views.sort(key=lambda x: x[1], reverse=True)
        
        # Select top views
        source_views = source_views[:5]  # Use at most 5 source views
        
        # For each source view, compute depth map
        depth_maps = []
        for src_id, _ in source_views:
            # Get source image
            src_image = images[src_id]
            if len(src_image.shape) > 2:
                src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
            else:
                src_image_gray = src_image
            
            # Compute disparity map
            # In a full implementation, this would use a proper stereo algorithm
            # For simplicity, we'll just use a basic block matcher
            stereo = cv2.StereoBM_create(numDisparities=16*8, blockSize=patch_size)
            disparity = stereo.compute(ref_image_gray, src_image_gray)
            
            # Convert to float and normalize
            disparity = disparity.astype(np.float32) / 16.0
            
            # Create mask for valid disparity
            mask = disparity > 0
            
            # Convert disparity to depth
            # In a proper implementation, this would use the camera parameters
            # For simplicity, we'll use a placeholder function
            depth = 100.0 / (disparity + 1e-6)  # Arbitrary scaling
            depth[~mask] = 0
            
            depth_maps.append(depth)
        
        # Combine depth maps (median)
        combined_depth = np.zeros((height, width), dtype=np.float32)
        valid_count = np.zeros((height, width), dtype=np.int32)
        
        for depth_map in depth_maps:
            mask = depth_map > 0
            combined_depth[mask] += depth_map[mask]
            valid_count[mask] += 1
        
        # Compute average depth
        mask = valid_count >= num_consistent_views
        combined_depth[mask] /= valid_count[mask]
        combined_depth[~mask] = 0
        
        # Create point cloud from depth map
        partial_cloud = pointcloud_from_depth(combined_depth, ref_camera, mask)
        
        # Assign colors from reference image
        if len(ref_image.shape) > 2:
            colors = np.asarray(ref_image)[np.where(mask)]
            if len(colors) == len(partial_cloud.points):
                partial_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Merge with result
        dense_cloud += partial_cloud
    
    # Remove duplicates
    dense_cloud = dense_cloud.voxel_down_sample(voxel_size=0.01)
    
    # Filter outliers
    dense_cloud = filter_pointcloud(dense_cloud)
    
    logger.info(f"Densification completed with {len(dense_cloud.points)} points")
    
    return dense_cloud


def voxel_downsample(pointcloud: o3d.geometry.PointCloud,
                    voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """Downsample point cloud using voxel grid.
    
    Args:
        pointcloud: Input point cloud
        voxel_size: Voxel size
        
    Returns:
        Downsampled point cloud
    """
    # Create a copy of the input point cloud
    result = o3d.geometry.PointCloud(pointcloud)
    
    # Apply voxel downsampling
    result = result.voxel_down_sample(voxel_size=voxel_size)
    
    logger.info(f"Downsampled point cloud from {len(pointcloud.points)} to {len(result.points)} points")
    
    return result


def poisson_reconstruction(pointcloud: o3d.geometry.PointCloud,
                          depth: int = 8,
                          width: int = 0,
                          scale: float = 1.1,
                          linear_fit: bool = False) -> o3d.geometry.TriangleMesh:
    """Reconstruct mesh from point cloud using Poisson reconstruction.
    
    Args:
        pointcloud: Input point cloud with normals
        depth: Maximum depth of octree
        width: Width for density filter
        scale: Scale factor for reconstruction
        linear_fit: Whether to use linear fit
        
    Returns:
        Reconstructed mesh
    """
    # Check if normals are available
    if not pointcloud.has_normals():
        logger.warning("Point cloud does not have normals, computing now")
        pointcloud = compute_normals(pointcloud)
    
    # Apply Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pointcloud, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
    
    logger.info(f"Poisson reconstruction created mesh with {len(mesh.triangles)} triangles")
    
    # Filter mesh based on density
    if width > 0:
        # Create density histogram
        densities = np.asarray(densities)
        density_colors = plt.cm.viridis(
            (densities - np.min(densities)) / (np.max(densities) - np.min(densities)))[:, :3]
        
        # Assign density colors to vertices
        mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        logger.info(f"Filtered mesh based on density, now has {len(mesh.triangles)} triangles")
    
    return mesh


def merge_pointclouds(pointclouds: List[o3d.geometry.PointCloud],
                     voxel_size: float = 0.01) -> o3d.geometry.PointCloud:
    """Merge multiple point clouds.
    
    Args:
        pointclouds: List of point clouds
        voxel_size: Voxel size for removing duplicates
        
    Returns:
        Merged point cloud
    """
    # Check if list is empty
    if not pointclouds:
        logger.warning("Empty list of point clouds")
        return o3d.geometry.PointCloud()
    
    # Create a copy of the first point cloud
    result = o3d.geometry.PointCloud(pointclouds[0])
    
    # Add other point clouds
    for i in range(1, len(pointclouds)):
        result += pointclouds[i]
    
    # Remove duplicates using voxel grid
    result = result.voxel_down_sample(voxel_size=voxel_size)
    
    logger.info(f"Merged {len(pointclouds)} point clouds, result has {len(result.points)} points")
    
    return result


def register_pointclouds(source: o3d.geometry.PointCloud,
                        target: o3d.geometry.PointCloud,
                        voxel_size: float = 0.05,
                        max_correspondence_distance: float = 0.1,
                        ransac_n: int = 4,
                        ransac_iter: int = 50000) -> np.ndarray:
    """Register two point clouds using global registration.
    
    Args:
        source: Source point cloud
        target: Target point cloud
        voxel_size: Voxel size for downsampling
        max_correspondence_distance: Maximum correspondence distance
        ransac_n: Number of points for RANSAC
        ransac_iter: Number of RANSAC iterations
        
    Returns:
        4x4 transformation matrix
    """
    # Downsample point clouds
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    # Compute FPFH features
    source_fpfh = compute_fpfh_features(compute_normals(source_down), radius=voxel_size * 5)
    target_fpfh = compute_fpfh_features(compute_normals(target_down), radius=voxel_size * 5)
    
    # Prepare RANSAC
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, 
        o3d.pipelines.registration.Feature(source_fpfh), 
        o3d.pipelines.registration.Feature(target_fpfh),
        max_correspondence_distance,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_correspondence_distance)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, 1000)
    )
    
    logger.info(f"Global registration with {ransac_result.fitness} fitness and {ransac_result.inlier_rmse} RMSE")
    
    # Refine registration
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    logger.info(f"ICP refinement with {icp_result.fitness} fitness and {icp_result.inlier_rmse} RMSE")
    
    return icp_result.transformation


def colorize_pointcloud(pointcloud: o3d.geometry.PointCloud,
                       images: List[np.ndarray],
                       cameras: Dict[int, Camera]) -> o3d.geometry.PointCloud:
    """Colorize point cloud using images.
    
    Args:
        pointcloud: Input point cloud
        images: List of input images
        cameras: Dictionary of cameras by image ID
        
    Returns:
        Colorized point cloud
    """
    # Create a copy of the input point cloud
    result = o3d.geometry.PointCloud(pointcloud)
    
    # Get points as numpy array
    points = np.asarray(result.points)
    
    # Initialize colors array
    colors = np.zeros((len(points), 3))
    weights = np.zeros(len(points))
    
    # Project points to each camera and collect colors
    for image_id, camera in cameras.items():
        # Skip if no image data
        if image_id >= len(images):
            continue
        
        image = images[image_id]
        if image is None:
            continue
        
        # Convert to BGR if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Project points to camera
        points_2d, depths = camera.project(points)
        
        # Iterate through projected points
        for i, (x, y) in enumerate(points_2d):
            # Check if point is inside image
            if 0 <= x < width and 0 <= y < height and depths[i] > 0:
                # Get color from image
                color = image[int(y), int(x)]
                
                # Weight by inverse depth
                weight = 1.0 / max(depths[i], 1e-6)
                
                # Update weighted color
                colors[i] += color * weight
                weights[i] += weight
    
    # Normalize colors
    mask = weights > 0
    colors[mask] = colors[mask] / weights[mask].reshape(-1, 1)
    
    # Set default color for points without weights
    colors[~mask] = [200, 200, 200]  # Default gray
    
    # Set colors
    result.colors = o3d.utility.Vector3dVector(colors / 255.0)
    
    logger.info(f"Colorized {np.sum(mask)} out of {len(points)} points")
    
    return result


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Point cloud processing")
    parser.add_argument("--input", required=True, help="Input point cloud file")
    parser.add_argument("--output", required=True, help="Output point cloud file")
    parser.add_argument("--filter", action="store_true", help="Apply filtering")
    parser.add_argument("--normals", action="store_true", help="Compute normals")
    parser.add_argument("--downsample", type=float, help="Voxel size for downsampling")
    
    args = parser.parse_args()
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(args.input)
    logger.info(f"Loaded point cloud with {len(pcd.points)} points")
    
    # Apply operations
    if args.filter:
        pcd = filter_pointcloud(pcd)
    
    if args.normals:
        pcd = compute_normals(pcd)
    
    if args.downsample:
        pcd = voxel_downsample(pcd, args.downsample)
    
    # Save result
    o3d.io.write_point_cloud(args.output, pcd)
    logger.info(f"Saved point cloud to {args.output}")