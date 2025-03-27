#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D transformation and geometry utility functions.

This module provides common utility functions for geometric transformations,
coordinate conversions, and normalization operations used throughout the
RECONTEXT project.

Author: James Wei
Date: 2024-01-15
Last modified: 2024-03-05
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize points to be centered at origin with unit std deviation.
    
    Args:
        points: Nx3 array of 3D points
        
    Returns:
        Tuple of (normalized points, center, scale)
    """
    # Compute center (mean)
    center = np.mean(points, axis=0)
    
    # Center points
    centered = points - center
    
    # Compute scale (standard deviation)
    scale = np.std(centered)
    
    # Handle degenerate case
    if scale < 1e-10:
        scale = 1.0
    
    # Normalize
    normalized = centered / scale
    
    return normalized, center, scale


def denormalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """Denormalize points using center and scale.
    
    Args:
        points: Nx3 array of normalized 3D points
        center: 3-element array of center coordinates
        scale: Scale factor
        
    Returns:
        Denormalized points
    """
    return points * scale + center


def normalize_pointcloud(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize point cloud to fit in a unit cube.
    
    Args:
        points: Nx3 array of 3D points
        
    Returns:
        Tuple of (normalized points, center, scale)
    """
    # Compute bounds
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # Compute center
    center = (min_bound + max_bound) / 2.0
    
    # Center points
    centered = points - center
    
    # Compute scale (maximum coordinate)
    scale = np.max(np.abs(centered))
    
    # Handle degenerate case
    if scale < 1e-10:
        scale = 1.0
    
    # Normalize to [-1, 1] cube
    normalized = centered / scale
    
    return normalized, center, scale


def create_rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix around X axis.
    
    Args:
        angle_deg: Angle in degrees
        
    Returns:
        3x3 rotation matrix
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a]
    ])
    
    return R


def create_rotation_matrix_y(angle_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix around Y axis.
    
    Args:
        angle_deg: Angle in degrees
        
    Returns:
        3x3 rotation matrix
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([
        [cos_a, 0.0, sin_a],
        [0.0, 1.0, 0.0],
        [-sin_a, 0.0, cos_a]
    ])
    
    return R


def create_rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """Create 3x3 rotation matrix around Z axis.
    
    Args:
        angle_deg: Angle in degrees
        
    Returns:
        3x3 rotation matrix
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a, cos_a, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    return R


def create_rotation_matrix(angles_deg: np.ndarray) -> np.ndarray:
    """Create 3x3 rotation matrix from Euler angles (XYZ order).
    
    Args:
        angles_deg: Array of 3 angles in degrees [X, Y, Z]
        
    Returns:
        3x3 rotation matrix
    """
    Rx = create_rotation_matrix_x(angles_deg[0])
    Ry = create_rotation_matrix_y(angles_deg[1])
    Rz = create_rotation_matrix_z(angles_deg[2])
    
    # Apply rotations in XYZ order
    R = Rz @ Ry @ Rx
    
    return R


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    
    return np.array([w, x, y, z])


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector to 3x3 rotation matrix.
    
    Args:
        rvec: 3-element Rodrigues rotation vector
        
    Returns:
        3x3 rotation matrix
    """
    import cv2
    R, _ = cv2.Rodrigues(rvec)
    return R


def rotation_matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Rodrigues rotation vector.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        3-element Rodrigues rotation vector
    """
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


def create_transformation_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 transformation matrix from rotation and translation.
    
    Args:
        R: 3x3 rotation matrix
        t: 3-element translation vector
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


def decompose_transformation_matrix(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose 4x4 transformation matrix into rotation and translation.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Tuple of (3x3 rotation matrix, 3-element translation vector)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def invert_transformation_matrix(T: np.ndarray) -> np.ndarray:
    """Invert 4x4 transformation matrix.
    
    Args:
        T: 4x4 transformation matrix
        
    Returns:
        Inverted 4x4 transformation matrix
    """
    R, t = decompose_transformation_matrix(T)
    R_inv = R.T
    t_inv = -R_inv @ t
    
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    
    return T_inv


def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Transform 3D points using a 4x4 transformation matrix.
    
    Args:
        points: Nx3 array of 3D points
        transformation: 4x4 transformation matrix
        
    Returns:
        Transformed points
    """
    # Extract rotation and translation
    R = transformation[:3, :3]
    t = transformation[:3, 3]
    
    # Apply transformation
    transformed = (R @ points.T).T + t
    
    return transformed


def world_to_camera(points: Union[np.ndarray, torch.Tensor], 
                   extrinsic: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Transform 3D points from world to camera coordinates.
    
    Args:
        points: Nx3 array of 3D points in world coordinates
        extrinsic: 4x4 camera extrinsic matrix
        
    Returns:
        Nx3 array of 3D points in camera coordinates
    """
    if isinstance(points, torch.Tensor):
        # PyTorch implementation
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Check if points need reshaping
        if len(points.shape) == 2:
            points_cam = torch.matmul(R, points.t()).t() + t.unsqueeze(0)
        else:
            # Handle batched points
            points_cam = torch.matmul(R, points.permute(0, 2, 1)).permute(0, 2, 1) + t.unsqueeze(0).unsqueeze(0)
            
        return points_cam
    else:
        # NumPy implementation
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Apply transformation
        points_cam = (R @ points.T).T + t
        
        return points_cam


def camera_to_world(points: Union[np.ndarray, torch.Tensor], 
                   extrinsic: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Transform 3D points from camera to world coordinates.
    
    Args:
        points: Nx3 array of 3D points in camera coordinates
        extrinsic: 4x4 camera extrinsic matrix
        
    Returns:
        Nx3 array of 3D points in world coordinates
    """
    if isinstance(points, torch.Tensor):
        # PyTorch implementation
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Invert transformation
        R_inv = R.transpose(0, 1)
        t_inv = -torch.matmul(R_inv, t)
        
        # Check if points need reshaping
        if len(points.shape) == 2:
            points_world = torch.matmul(R_inv, (points - t.unsqueeze(0)).t()).t()
        else:
            # Handle batched points
            points_world = torch.matmul(R_inv, (points - t.unsqueeze(0).unsqueeze(0)).permute(0, 2, 1)).permute(0, 2, 1)
            
        return points_world
    else:
        # NumPy implementation
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        
        # Invert transformation
        R_inv = R.T
        t_inv = -R_inv @ t
        
        # Apply inverse transformation
        points_world = (R_inv @ (points - t).T).T
        
        return points_world


def camera_to_pixel(points_cam: Union[np.ndarray, torch.Tensor], 
                   intrinsic: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Project 3D points in camera coordinates to pixel coordinates.
    
    Args:
        points_cam: Nx3 array of 3D points in camera coordinates
        intrinsic: 3x3 camera intrinsic matrix
        
    Returns:
        Nx2 array of pixel coordinates
    """
    if isinstance(points_cam, torch.Tensor):
        # PyTorch implementation
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        # Avoid division by zero
        z = torch.clamp(z, min=1e-6)
        
        # Project to image plane
        u = intrinsic[0, 0] * (x / z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (y / z) + intrinsic[1, 2]
        
        return torch.stack([u, v], dim=1)
    else:
        # NumPy implementation
        x = points_cam[:, 0]
        y = points_cam[:, 1]
        z = points_cam[:, 2]
        
        # Avoid division by zero
        z = np.maximum(z, 1e-6)
        
        # Project to image plane
        u = intrinsic[0, 0] * (x / z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (y / z) + intrinsic[1, 2]
        
        return np.stack([u, v], axis=1)


def pixel_to_ray(pixels: Union[np.ndarray, torch.Tensor], 
                intrinsic: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Convert pixel coordinates to 3D rays in camera coordinates.
    
    Args:
        pixels: Nx2 array of pixel coordinates
        intrinsic: 3x3 camera intrinsic matrix
        
    Returns:
        Nx3 array of normalized 3D rays in camera coordinates
    """
    if isinstance(pixels, torch.Tensor):
        # PyTorch implementation
        u = pixels[:, 0]
        v = pixels[:, 1]
        
        # Unproject to normalized image coordinates
        x = (u - intrinsic[0, 2]) / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) / intrinsic[1, 1]
        
        # Create ray direction
        rays = torch.stack([x, y, torch.ones_like(x)], dim=1)
        
        # Normalize rays
        norm = torch.norm(rays, dim=1, keepdim=True)
        rays = rays / norm
        
        return rays
    else:
        # NumPy implementation
        u = pixels[:, 0]
        v = pixels[:, 1]
        
        # Unproject to normalized image coordinates
        x = (u - intrinsic[0, 2]) / intrinsic[0, 0]
        y = (v - intrinsic[1, 2]) / intrinsic[1, 1]
        
        # Create ray direction
        rays = np.stack([x, y, np.ones_like(x)], axis=1)
        
        # Normalize rays
        norm = np.linalg.norm(rays, axis=1, keepdims=True)
        rays = rays / norm
        
        return rays


def triangulate_rays(origins: np.ndarray, 
                    directions: np.ndarray,
                    method: str = 'linear') -> np.ndarray:
    """Triangulate 3D point from multiple rays.
    
    Args:
        origins: Nx3 array of ray origins
        directions: Nx3 array of ray directions (normalized)
        method: Triangulation method ('linear' or 'nonlinear')
        
    Returns:
        3D point
    """
    # Check inputs
    if len(origins) < 2 or len(directions) < 2:
        logger.warning("At least 2 rays needed for triangulation")
        return None
    
    if origins.shape != directions.shape:
        logger.error("Origins and directions must have the same shape")
        return None
    
    # Implement linear method (closest point to multiple lines)
    if method == 'linear':
        A = np.zeros((3 * len(origins), 3))
        b = np.zeros(3 * len(origins))
        
        for i in range(len(origins)):
            origin = origins[i]
            direction = directions[i]
            
            # Create cross product matrix
            cross_matrix = np.array([
                [0, -direction[2], direction[1]],
                [direction[2], 0, -direction[0]],
                [-direction[1], direction[0], 0]
            ])
            
            A[3*i:3*(i+1)] = cross_matrix
            b[3*i:3*(i+1)] = cross_matrix @ origin
        
        # Solve least squares problem
        point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return point
    
    # Implement nonlinear method
    elif method == 'nonlinear':
        # This would implement a nonlinear optimization to minimize reprojection error
        # For simplicity, we fall back to linear method
        logger.warning("Nonlinear triangulation not implemented, using linear method")
        return triangulate_rays(origins, directions, method='linear')
    
    else:
        logger.error(f"Unknown triangulation method: {method}")
        return None


def compute_triangulation_angle(camera1_pos: np.ndarray, 
                               camera2_pos: np.ndarray,
                               point: np.ndarray) -> float:
    """Compute triangulation angle between two cameras and a 3D point.
    
    Args:
        camera1_pos: Position of first camera
        camera2_pos: Position of second camera
        point: 3D point
        
    Returns:
        Triangulation angle in degrees
    """
    # Compute vectors from cameras to point
    vec1 = point - camera1_pos
    vec2 = point - camera2_pos
    
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Compute angle
    cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_epipolar_error(point1: np.ndarray, 
                          point2: np.ndarray,
                          fundamental_matrix: np.ndarray) -> float:
    """Compute epipolar constraint error for a point correspondence.
    
    Args:
        point1: 2D point in first image
        point2: 2D point in second image
        fundamental_matrix: 3x3 fundamental matrix
        
    Returns:
        Epipolar constraint error
    """
    # Convert to homogeneous coordinates
    p1 = np.array([point1[0], point1[1], 1.0])
    p2 = np.array([point2[0], point2[1], 1.0])
    
    # Compute epipolar constraint
    error = np.abs(p2.dot(fundamental_matrix).dot(p1))
    
    return error


def compute_reprojection_error(point_3d: np.ndarray,
                              point_2d: np.ndarray,
                              projection_matrix: np.ndarray) -> float:
    """Compute reprojection error for a 3D-2D correspondence.
    
    Args:
        point_3d: 3D point
        point_2d: 2D point
        projection_matrix: 3x4 projection matrix
        
    Returns:
        Reprojection error in pixels
    """
    # Convert to homogeneous coordinates
    p3d = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    
    # Project 3D point
    p = projection_matrix @ p3d
    p /= p[2]  # Divide by z
    projected_2d = p[:2]
    
    # Compute error
    error = np.linalg.norm(projected_2d - point_2d)
    
    return error


def rigid_transform_3d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find optimal rigid transformation between two point sets.
    
    Implements the Kabsch algorithm.
    
    Args:
        A: Nx3 array of source points
        B: Nx3 array of target points
        
    Returns:
        Tuple of (3x3 rotation matrix, 3-element translation vector)
    """
    assert A.shape == B.shape
    
    # Find centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # Center points
    AA = A - centroid_A
    BB = B - centroid_B
    
    # Compute covariance matrix
    H = AA.T @ BB
    
    # Find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Find translation
    t = centroid_B - R @ centroid_A
    
    return R, t


def icp_registration(source: np.ndarray, 
                    target: np.ndarray,
                    max_iterations: int = 20,
                    tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
    """Perform Iterative Closest Point registration.
    
    Args:
        source: Nx3 array of source points
        target: Mx3 array of target points
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (4x4 transformation matrix, RMS error)
    """
    # Create a copy of source
    current = source.copy()
    prev_error = np.inf
    
    # Build KD-tree for target
    from scipy.spatial import cKDTree
    target_tree = cKDTree(target)
    
    # Identity transformation
    T = np.eye(4)
    
    for i in range(max_iterations):
        # Find closest points
        distances, indices = target_tree.query(current, k=1)
        correspondence = target[indices]
        
        # Compute transformation
        R, t = rigid_transform_3d(current, correspondence)
        
        # Update transformation
        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3] = t
        T = T_step @ T
        
        # Apply transformation
        current = (R @ current.T).T + t
        
        # Compute error
        mean_error = np.mean(distances)
        
        # Check convergence
        if np.abs(prev_error - mean_error) < tolerance:
            break
            
        prev_error = mean_error
    
    return T, prev_error


if __name__ == "__main__":
    # Test normalization
    points = np.random.rand(100, 3) * 10 - 5
    normalized, center, scale = normalize_points(points)
    denormalized = denormalize_points(normalized, center, scale)
    
    assert np.allclose(points, denormalized)
    print("Normalization test passed")
    
    # Test rigid transformation
    A = np.random.rand(10, 3)
    R_true = create_rotation_matrix(np.array([30, 45, 60]))
    t_true = np.array([1, 2, 3])
    B = (R_true @ A.T).T + t_true
    
    R_est, t_est = rigid_transform_3d(A, B)
    
    assert np.allclose(R_true, R_est, atol=1e-5)
    assert np.allclose(t_true, t_est, atol=1e-5)
    print("Rigid transformation test passed")