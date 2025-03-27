#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera module handling camera models, calibration, and pose estimation.
Supports multiple camera models including perspective and fisheye.

Author: Sarah Li
Date: 2024-01-20
Last modified: 2024-03-05
"""

import numpy as np
import cv2
import json
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    width: int
    height: int
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion 1
    k2: float = 0.0  # Radial distortion 2
    p1: float = 0.0  # Tangential distortion 1
    p2: float = 0.0  # Tangential distortion 2
    k3: float = 0.0  # Radial distortion 3
    model: str = "perspective"  # Camera model
    
    @property
    def K(self) -> np.ndarray:
        """Get intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
    
    @property
    def distortion(self) -> np.ndarray:
        """Get distortion coefficients."""
        if self.model == "fisheye":
            return np.array([self.k1, self.k2, self.p1, self.p2])
        else:
            return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "k1": self.k1,
            "k2": self.k2,
            "p1": self.p1,
            "p2": self.p2,
            "k3": self.k3,
            "model": self.model
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraIntrinsics':
        """Create from dictionary."""
        return cls(
            width=data.get("width", 0),
            height=data.get("height", 0),
            fx=data.get("fx", 0.0),
            fy=data.get("fy", 0.0),
            cx=data.get("cx", 0.0),
            cy=data.get("cy", 0.0),
            k1=data.get("k1", 0.0),
            k2=data.get("k2", 0.0),
            p1=data.get("p1", 0.0),
            p2=data.get("p2", 0.0),
            k3=data.get("k3", 0.0),
            model=data.get("model", "perspective")
        )
    
    @classmethod
    def from_calibration_matrix(cls, K: np.ndarray, width: int, height: int,
                              distortion: Optional[np.ndarray] = None,
                              model: str = "perspective") -> 'CameraIntrinsics':
        """Create from calibration matrix."""
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        if distortion is None:
            distortion = np.zeros(5)
            
        k1 = distortion[0] if len(distortion) > 0 else 0.0
        k2 = distortion[1] if len(distortion) > 1 else 0.0
        p1 = distortion[2] if len(distortion) > 2 else 0.0
        p2 = distortion[3] if len(distortion) > 3 else 0.0
        k3 = distortion[4] if len(distortion) > 4 else 0.0
        
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            k1=k1,
            k2=k2,
            p1=p1,
            p2=p2,
            k3=k3,
            model=model
        )
    
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort 2D points.
        
        Args:
            points: Nx2 array of distorted points
            
        Returns:
            Nx2 array of undistorted points
        """
        # Reshape for OpenCV
        points_reshaped = points.reshape(-1, 1, 2)
        
        # Get camera matrix and distortion coefficients
        camera_matrix = self.K
        dist_coeffs = self.distortion
        
        # Undistort points
        if self.model == "fisheye":
            undistorted_points = cv2.fisheye.undistortPoints(
                points_reshaped, camera_matrix, dist_coeffs, None, camera_matrix)
        else:
            undistorted_points = cv2.undistortPoints(
                points_reshaped, camera_matrix, dist_coeffs, None, camera_matrix)
        
        # Reshape back
        return undistorted_points.reshape(-1, 2)
    
    def distort_points(self, points: np.ndarray) -> np.ndarray:
        """Distort 2D points (inverse of undistort_points).
        
        This is more complex as OpenCV doesn't provide a direct function.
        We use an approximation by projecting 3D points.
        
        Args:
            points: Nx2 array of undistorted points
            
        Returns:
            Nx2 array of distorted points
        """
        # Convert to normalized camera coordinates
        normalized_points = np.zeros((len(points), 3))
        normalized_points[:, :2] = points
        normalized_points[:, 2] = 1.0
        
        # Project points
        if self.model == "fisheye":
            distorted_points, _ = cv2.fisheye.projectPoints(
                normalized_points[:, :3], np.zeros(3), np.zeros(3), 
                self.K, self.distortion)
        else:
            distorted_points, _ = cv2.projectPoints(
                normalized_points[:, :3], np.zeros(3), np.zeros(3), 
                self.K, self.distortion)
        
        # Reshape
        return distorted_points.reshape(-1, 2)


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (pose)."""
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :3] = self.R
        matrix[:3, 3] = self.t.flatten()
        return matrix
    
    @property
    def inverse(self) -> 'CameraExtrinsics':
        """Get inverse transformation."""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return CameraExtrinsics(R_inv, t_inv)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "R": self.R.tolist(),
            "t": self.t.flatten().tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CameraExtrinsics':
        """Create from dictionary."""
        R = np.array(data.get("R", np.eye(3).tolist()))
        t = np.array(data.get("t", np.zeros(3).tolist())).reshape(3, 1)
        return cls(R, t)
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> 'CameraExtrinsics':
        """Create from 4x4 transformation matrix."""
        R = matrix[:3, :3]
        t = matrix[:3, 3].reshape(3, 1)
        return cls(R, t)
    
    @classmethod
    def from_quaternion(cls, q: np.ndarray, t: np.ndarray) -> 'CameraExtrinsics':
        """Create from quaternion and translation vector."""
        # Convert quaternion to rotation matrix
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        return cls(R, t.reshape(3, 1))


class Camera:
    """Full camera model with intrinsics and extrinsics."""
    
    def __init__(self, 
                 intrinsics: CameraIntrinsics,
                 extrinsics: Optional[CameraExtrinsics] = None):
        """Initialize camera.
        
        Args:
            intrinsics: Camera intrinsics
            extrinsics: Camera extrinsics (pose)
        """
        self.intrinsics = intrinsics
        
        if extrinsics is None:
            # Identity transform (camera at origin, looking down Z axis)
            self.extrinsics = CameraExtrinsics(
                R=np.eye(3),
                t=np.zeros((3, 1))
            )
        else:
            self.extrinsics = extrinsics
    
    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: Nx3 array of 3D points in world coordinates
            
        Returns:
            Tuple of (Nx2 array of 2D points, Nx1 array of depths)
        """
        # Convert to camera coordinates
        R = self.extrinsics.R
        t = self.extrinsics.t
        
        # Transform points to camera coordinates
        points_camera = (R @ points_3d.T + t).T
        
        # Get depths
        depths = points_camera[:, 2]
        
        # Project to image plane
        if self.intrinsics.model == "fisheye":
            # For fisheye model
            points_2d, _ = cv2.fisheye.projectPoints(
                points_camera, np.zeros(3), np.zeros(3),
                self.intrinsics.K, self.intrinsics.distortion)
            points_2d = points_2d.reshape(-1, 2)
        else:
            # For perspective model
            points_2d, _ = cv2.projectPoints(
                points_camera, np.zeros(3), np.zeros(3),
                self.intrinsics.K, self.intrinsics.distortion)
            points_2d = points_2d.reshape(-1, 2)
        
        return points_2d, depths
    
    def backproject(self, points_2d: np.ndarray, depth: Union[float, np.ndarray]) -> np.ndarray:
        """Backproject 2D image points to 3D camera coordinates.
        
        Args:
            points_2d: Nx2 array of 2D points in image coordinates
            depth: Depth value(s) for the points
            
        Returns:
            Nx3 array of 3D points in world coordinates
        """
        # Undistort points
        points_undistorted = self.intrinsics.undistort_points(points_2d)
        
        # Convert to normalized camera coordinates
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy
        
        # Normalize coordinates
        x_normalized = (points_undistorted[:, 0] - cx) / fx
        y_normalized = (points_undistorted[:, 1] - cy) / fy
        
        # Create 3D points in camera coordinates
        if isinstance(depth, np.ndarray):
            # Array of depths
            points_camera = np.column_stack([
                x_normalized * depth,
                y_normalized * depth,
                depth
            ])
        else:
            # Single depth value
            points_camera = np.column_stack([
                x_normalized * depth,
                y_normalized * depth,
                np.ones_like(x_normalized) * depth
            ])
        
        # Transform to world coordinates
        R_inv = self.extrinsics.R.T
        t = self.extrinsics.t
        points_world = (R_inv @ (points_camera.T - t)).T
        
        return points_world
    
    def get_ray(self, point_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get ray from camera center through image point.
        
        Args:
            point_2d: 2D point in image coordinates
            
        Returns:
            Tuple of (ray origin, ray direction)
        """
        # Undistort point
        point_undistorted = self.intrinsics.undistort_points(point_2d.reshape(1, 2)).flatten()
        
        # Compute ray in camera coordinates
        fx = self.intrinsics.fx
        fy = self.intrinsics.fy
        cx = self.intrinsics.cx
        cy = self.intrinsics.cy
        
        x_normalized = (point_undistorted[0] - cx) / fx
        y_normalized = (point_undistorted[1] - cy) / fy
        
        # Ray direction in camera coordinates
        ray_dir_camera = np.array([x_normalized, y_normalized, 1.0])
        ray_dir_camera = ray_dir_camera / np.linalg.norm(ray_dir_camera)
        
        # Transform to world coordinates
        R = self.extrinsics.R
        t = self.extrinsics.t
        
        # Ray origin is camera center in world coordinates
        ray_origin = -R.T @ t
        
        # Ray direction in world coordinates
        ray_dir = R.T @ ray_dir_camera
        
        return ray_origin.flatten(), ray_dir
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "intrinsics": self.intrinsics.to_dict(),
            "extrinsics": self.extrinsics.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Camera':
        """Create from dictionary."""
        intrinsics = CameraIntrinsics.from_dict(data.get("intrinsics", {}))
        extrinsics = CameraExtrinsics.from_dict(data.get("extrinsics", {}))
        return cls(intrinsics, extrinsics)
    
    def save(self, filepath: str):
        """Save camera parameters to file.
        
        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Camera':
        """Load camera parameters from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Camera object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


def estimate_camera_pose(points_3d: np.ndarray, points_2d: np.ndarray, 
                        intrinsics: CameraIntrinsics, method: str = "pnp") -> CameraExtrinsics:
    """Estimate camera pose from 3D-2D correspondences.
    
    Args:
        points_3d: Nx3 array of 3D points in world coordinates
        points_2d: Nx2 array of corresponding 2D points in image coordinates
        intrinsics: Camera intrinsics
        method: PnP method ('pnp', 'epnp', 'dls', 'upnp')
        
    Returns:
        Camera extrinsics (pose)
    """
    # Select PnP method
    if method == 'epnp':
        pnp_method = cv2.SOLVEPNP_EPNP
    elif method == 'dls':
        pnp_method = cv2.SOLVEPNP_DLS
    elif method == 'upnp':
        pnp_method = cv2.SOLVEPNP_UPNP
    else:
        pnp_method = cv2.SOLVEPNP_ITERATIVE
    
    # Get camera matrix and distortion coefficients
    camera_matrix = intrinsics.K
    dist_coeffs = intrinsics.distortion
    
    # Solve PnP
    if intrinsics.model == "fisheye":
        # For fisheye cameras, first undistort points
        points_2d_undistorted = intrinsics.undistort_points(points_2d)
        
        # Then solve with zero distortion
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d_undistorted, camera_matrix, None, flags=pnp_method)
    else:
        # Standard case
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, camera_matrix, dist_coeffs, flags=pnp_method)
    
    if not success:
        raise ValueError("Failed to estimate camera pose")
    
    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    
    return CameraExtrinsics(R, tvec)


def calibrate_camera(points_3d: List[np.ndarray], 
                    points_2d: List[np.ndarray],
                    image_size: Tuple[int, int],
                    model: str = "perspective") -> CameraIntrinsics:
    """Calibrate camera from multiple 3D-2D correspondences.
    
    Args:
        points_3d: List of Nx3 arrays of 3D points
        points_2d: List of Nx2 arrays of corresponding 2D points
        image_size: (width, height) of the image
        model: Camera model ('perspective' or 'fisheye')
        
    Returns:
        Camera intrinsics
    """
    width, height = image_size
    
    if model == "fisheye":
        # Fisheye calibration
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_FIX_SKEW +
            cv2.fisheye.CALIB_CHECK_COND
        )
        
        # Prepare data
        object_points = [pts.reshape(-1, 1, 3) for pts in points_3d]
        image_points = [pts.reshape(-1, 1, 2) for pts in points_2d]
        
        # Initialize camera matrix
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        
        # Calibrate
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points, image_points, (width, height), K, D,
            flags=calibration_flags)
        
        # Create intrinsics
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            k1=D[0, 0],
            k2=D[1, 0],
            p1=D[2, 0],
            p2=D[3, 0],
            model="fisheye"
        )
        
    else:
        # Perspective calibration
        calibration_flags = (
            cv2.CALIB_RATIONAL_MODEL +
            cv2.CALIB_FIX_PRINCIPAL_POINT +
            cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        # Prepare data
        object_points = [pts.reshape(-1, 1, 3) for pts in points_3d]
        image_points = [pts.reshape(-1, 1, 2) for pts in points_2d]
        
        # Initialize camera matrix
        K = np.zeros((3, 3))
        K[0, 0] = width  # Rough guess for focal length
        K[1, 1] = width
        K[0, 2] = width / 2
        K[1, 2] = height / 2
        K[2, 2] = 1
        
        D = np.zeros((5, 1))
        
        # Calibrate
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, (width, height), K, D,
            flags=calibration_flags)
        
        # Create intrinsics
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            k1=D[0, 0],
            k2=D[1, 0],
            p1=D[2, 0],
            p2=D[3, 0],
            k3=D[4, 0],
            model="perspective"
        )
    
    return intrinsics


def triangulate_point(cameras: List[Camera], 
                     points_2d: List[np.ndarray],
                     min_angle_deg: float = 3.0) -> Optional[np.ndarray]:
    """Triangulate 3D point from multiple views.
    
    Args:
        cameras: List of cameras
        points_2d: List of corresponding 2D points in each view
        min_angle_deg: Minimum triangulation angle in degrees
        
    Returns:
        3D point or None if triangulation fails
    """
    if len(cameras) < 2 or len(points_2d) < 2:
        return None
    
    # Check number of views
    if len(cameras) != len(points_2d):
        raise ValueError("Number of cameras and 2D points must match")
    
    # Prepare projection matrices
    P_matrices = []
    for camera in cameras:
        K = camera.intrinsics.K
        R = camera.extrinsics.R
        t = camera.extrinsics.t
        
        # Projection matrix: K @ [R|t]
        P = K @ np.hstack((R, t))
        P_matrices.append(P)
    
    # Undistort points if needed
    undistorted_points = []
    for camera, point in zip(cameras, points_2d):
        if camera.intrinsics.model == "fisheye" or np.any(camera.intrinsics.distortion != 0):
            # Need to undistort
            undistorted = camera.intrinsics.undistort_points(point.reshape(1, 2))
            undistorted_points.append(undistorted.flatten())
        else:
            # Already undistorted
            undistorted_points.append(point)
    
    # Check triangulation angles
    sufficient_angle = False
    for i in range(len(cameras)):
        for j in range(i+1, len(cameras)):
            # Get camera centers
            C1 = -cameras[i].extrinsics.R.T @ cameras[i].extrinsics.t
            C2 = -cameras[j].extrinsics.R.T @ cameras[j].extrinsics.t
            
            # Get viewing rays
            ray1 = cameras[i].get_ray(points_2d[i])[1]
            ray2 = cameras[j].get_ray(points_2d[j])[1]
            
            # Compute angle between rays
            angle = np.arccos(np.clip(np.dot(ray1, ray2), -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            if angle_deg >= min_angle_deg:
                sufficient_angle = True
                break
        
        if sufficient_angle:
            break
    
    if not sufficient_angle:
        # Insufficient triangulation angle
        return None
    
    # Triangulate using DLT algorithm
    A = np.zeros((len(cameras) * 2, 4))
    
    for i, (P, point) in enumerate(zip(P_matrices, undistorted_points)):
        x, y = point
        A[i*2] = x * P[2] - P[0]
        A[i*2+1] = y * P[2] - P[1]
    
    # Solve system A * X = 0
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Convert to inhomogeneous coordinates
    X = X / X[3]
    point_3d = X[:3]
    
    return point_3d


def estimate_camera_intrinsics(focal_length_px: Optional[float] = None,
                             image_size: Optional[Tuple[int, int]] = None) -> CameraIntrinsics:
    """Estimate camera intrinsics from focal length or image size.
    
    Args:
        focal_length_px: Focal length in pixels
        image_size: (width, height) of the image
        
    Returns:
        Camera intrinsics
    """
    if image_size is None:
        raise ValueError("Image size is required")
        
    width, height = image_size
    
    if focal_length_px is None:
        # Estimate focal length from image size (common heuristic)
        focal_length_px = max(width, height) * 1.2
    
    # Principal point at image center
    cx = width / 2
    cy = height / 2
    
    # Create intrinsics
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=focal_length_px,
        fy=focal_length_px,
        cx=cx,
        cy=cy,
        model="perspective"
    )
    
    return intrinsics


def relative_pose_from_essential(E: np.ndarray, 
                                points_2d_1: np.ndarray,
                                points_2d_2: np.ndarray,
                                K1: np.ndarray,
                                K2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract relative pose from essential matrix with cheirality check.
    
    Args:
        E: Essential matrix
        points_2d_1: 2D points in first view
        points_2d_2: 2D points in second view
        K1: Intrinsic matrix of first camera
        K2: Intrinsic matrix of second camera
        
    Returns:
        Tuple of (R, t) where R is rotation matrix and t is translation vector
    """
    # Decompose essential matrix
    U, _, Vt = np.linalg.svd(E)
    
    # Create W matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Ensure rotations have det(R) = 1
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    # Two possible translations
    t1 = U[:, 2]
    t2 = -U[:, 2]
    
    # Four possible solutions
    solutions = [
        (R1, t1.reshape(3, 1)),
        (R1, t2.reshape(3, 1)),
        (R2, t1.reshape(3, 1)),
        (R2, t2.reshape(3, 1))
    ]
    
    # Normalize coordinates
    points_2d_1_norm = cv2.undistortPoints(points_2d_1.reshape(-1, 1, 2), K1, None)
    points_2d_2_norm = cv2.undistortPoints(points_2d_2.reshape(-1, 1, 2), K2, None)
    
    # Check cheirality constraint (points must be in front of both cameras)
    max_positive = 0
    best_solution = None
    
    for R, t in solutions:
        # First camera: [I|0]
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        
        # Second camera: [R|t]
        P2 = np.hstack((R, t))
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(
            P1, P2, points_2d_1_norm.reshape(-1, 2).T, points_2d_2_norm.reshape(-1, 2).T)
        
        # Convert to inhomogeneous coordinates
        points_3d = (points_4d / points_4d[3]).T[:, :3]
        
        # Transform to second camera
        points_3d_cam2 = (R @ points_3d.T + t).T
        
        # Count points with positive depth in both cameras
        positive_z1 = points_3d[:, 2] > 0
        positive_z2 = points_3d_cam2[:, 2] > 0
        positive_count = np.sum(positive_z1 & positive_z2)
        
        if positive_count > max_positive:
            max_positive = positive_count
            best_solution = (R, t)
    
    if best_solution is None:
        raise ValueError("Could not find a valid solution")
    
    return best_solution


def decompose_projection_matrix(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose projection matrix into K, R, t.
    
    Args:
        P: 3x4 projection matrix
        
    Returns:
        Tuple of (K, R, t)
    """
    # Extract 3x3 part
    M = P[:, :3]
    
    # RQ decomposition
    K, R = np.linalg.qr(M.T)
    K = K.T
    R = R.T
    
    # Ensure K has positive diagonal
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R
    
    # Ensure R is a rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        R = -R
    
    # Extract translation
    t = np.linalg.inv(K) @ P[:, 3]
    
    # Normalize K
    K = K / K[2, 2]
    
    return K, R, t


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera module example")
    parser.add_argument("--image1", help="First image")
    parser.add_argument("--image2", help="Second image")
    parser.add_argument("--focal_length", type=float, default=1000.0, help="Focal length in pixels")
    
    args = parser.parse_args()
    
    if args.image1 and args.image2:
        # Load images
        img1 = cv2.imread(args.image1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(args.image2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print("Failed to load images")
            return
        
        # Find keypoints and matches
        detector = cv2.SIFT_create()
        kp1, desc1 = detector.detectAndCompute(img1, None)
        kp2, desc2 = detector.detectAndCompute(img2, None)
        
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # Estimate intrinsics
        h, w = img1.shape
        intrinsics = estimate_camera_intrinsics(args.focal_length, (w, h))
        K = intrinsics.K
        
        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Extract relative pose
        R, t = relative_pose_from_essential(E, pts1, pts2, K, K)
        
        # Create camera objects
        camera1 = Camera(intrinsics)
        camera2 = Camera(intrinsics, CameraExtrinsics(R, t))
        
        # Triangulate points
        inliers = mask.ravel() == 1
        points_3d = []
        
        for i in range(sum(inliers)):
            point = triangulate_point([camera1, camera2], [pts1[i], pts2[i]])
            if point is not None:
                points_3d.append(point)
        
        print(f"Triangulated {len(points_3d)} points")
        
        # Save cameras
        camera1.save("camera1.json")
        camera2.save("camera2.json")
        
        print("Cameras saved to camera1.json and camera2.json")
    else:
        print("Please provide two images")


if __name__ == "__main__":
    main()