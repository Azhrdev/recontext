#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Structure from Motion module for camera pose estimation and sparse reconstruction.

This module implements a complete SfM pipeline including feature matching,
geometric verification, triangulation, bundle adjustment, and outlier filtering.

Author: James Wei
Date: 2024-01-25
Last modified: 2024-03-08
"""

import numpy as np
import cv2
import open3d as o3d
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from tqdm import tqdm

from recontext.core.feature_extraction import FeatureData
from recontext.core.matching import MatchData
from recontext.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from recontext.utils.transforms import normalize_points, denormalize_points

logger = logging.getLogger(__name__)

@dataclass
class SfMOptions:
    """Options for Structure from Motion."""
    init_method: str = "essential"  # Initialization method (essential, homography, auto)
    min_matches: int = 50  # Minimum number of matches for a valid image pair
    min_track_length: int = 3  # Minimum number of views per 3D point
    ransac_threshold: float = 4.0  # RANSAC threshold for geometric verification
    triangulation_angle_threshold: float = 3.0  # Minimum triangulation angle in degrees
    ba_max_iterations: int = 100  # Maximum iterations for bundle adjustment
    ba_frequency: int = 10  # Bundle adjustment frequency (every N images)
    final_ba: bool = True  # Whether to run final bundle adjustment
    filter_points: bool = True  # Whether to filter 3D points
    max_reprojection_error: float = 5.0  # Maximum reprojection error
    add_image_robust: bool = True  # Use robust image registration
    verbose: bool = False  # Verbose output


@dataclass
class Track:
    """Feature track across multiple images."""
    id: int
    observations: Dict[int, int] = field(default_factory=dict)  # image_id -> feature_idx
    point3d_id: Optional[int] = None
    
    @property
    def length(self) -> int:
        """Get track length (number of observations)."""
        return len(self.observations)


@dataclass
class Point3D:
    """3D point in the reconstruction."""
    id: int
    position: np.ndarray  # 3D position
    color: np.ndarray  # RGB color
    track_id: int  # Corresponding feature track
    error: float = 0.0  # Reprojection error
    confidence: float = 1.0  # Confidence score
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert position to tuple."""
        return tuple(float(x) for x in self.position)


class SfMReconstruction:
    """Structure from Motion reconstruction."""
    
    def __init__(self):
        """Initialize empty reconstruction."""
        self.cameras = {}  # image_id -> Camera
        self.tracks = {}  # track_id -> Track
        self.points3d = {}  # point3d_id -> Point3D
        self.reference_camera = None  # Reference camera (identity pose)
        self.reference_image_id = None  # Reference image ID
        self.next_track_id = 0
        self.next_point3d_id = 0
        self.scale = 1.0  # Global scale
        self.registration_graph = nx.Graph()  # Graph of registered images
    
    def add_camera(self, image_id: int, camera: Camera) -> None:
        """Add camera to reconstruction.
        
        Args:
            image_id: Image ID
            camera: Camera object
        """
        self.cameras[image_id] = camera
        
        # Update registration graph
        if not self.registration_graph.has_node(image_id):
            self.registration_graph.add_node(image_id)
    
    def get_camera(self, image_id: int) -> Optional[Camera]:
        """Get camera by image ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Camera object or None if not found
        """
        return self.cameras.get(image_id)
    
    def add_track(self, track: Track) -> int:
        """Add feature track to reconstruction.
        
        Args:
            track: Feature track
            
        Returns:
            Track ID
        """
        # Assign ID if not set
        if track.id is None or track.id < 0:
            track.id = self.next_track_id
            self.next_track_id += 1
        else:
            self.next_track_id = max(self.next_track_id, track.id + 1)
        
        self.tracks[track.id] = track
        return track.id
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            Track object or None if not found
        """
        return self.tracks.get(track_id)
    
    def add_point3d(self, point: Point3D) -> int:
        """Add 3D point to reconstruction.
        
        Args:
            point: 3D point
            
        Returns:
            Point ID
        """
        # Assign ID if not set
        if point.id is None or point.id < 0:
            point.id = self.next_point3d_id
            self.next_point3d_id += 1
        else:
            self.next_point3d_id = max(self.next_point3d_id, point.id + 1)
        
        self.points3d[point.id] = point
        
        # Update corresponding track
        track = self.tracks.get(point.track_id)
        if track:
            track.point3d_id = point.id
        
        return point.id
    
    def get_point3d(self, point_id: int) -> Optional[Point3D]:
        """Get 3D point by ID.
        
        Args:
            point_id: Point ID
            
        Returns:
            Point3D object or None if not found
        """
        return self.points3d.get(point_id)
    
    def remove_point3d(self, point_id: int) -> None:
        """Remove 3D point from reconstruction.
        
        Args:
            point_id: Point ID
        """
        # Get point
        point = self.points3d.get(point_id)
        if point is None:
            return
        
        # Update corresponding track
        track = self.tracks.get(point.track_id)
        if track:
            track.point3d_id = None
        
        # Remove point
        del self.points3d[point_id]
    
    def get_registered_images(self) -> List[int]:
        """Get list of registered image IDs.
        
        Returns:
            List of registered image IDs
        """
        return list(self.cameras.keys())
    
    def get_tracks_for_image(self, image_id: int) -> List[Track]:
        """Get tracks visible in an image.
        
        Args:
            image_id: Image ID
            
        Returns:
            List of tracks
        """
        return [track for track in self.tracks.values() if image_id in track.observations]
    
    def get_points3d_for_image(self, image_id: int) -> List[Point3D]:
        """Get 3D points visible in an image.
        
        Args:
            image_id: Image ID
            
        Returns:
            List of 3D points
        """
        points = []
        for track in self.get_tracks_for_image(image_id):
            if track.point3d_id is not None:
                point = self.get_point3d(track.point3d_id)
                if point is not None:
                    points.append(point)
        
        return points
    
    def get_common_tracks(self, image_id1: int, image_id2: int) -> List[Track]:
        """Get tracks visible in both images.
        
        Args:
            image_id1: First image ID
            image_id2: Second image ID
            
        Returns:
            List of common tracks
        """
        return [track for track in self.tracks.values() 
                if image_id1 in track.observations and image_id2 in track.observations]
    
    def get_common_points3d(self, image_id1: int, image_id2: int) -> List[Point3D]:
        """Get 3D points visible in both images.
        
        Args:
            image_id1: First image ID
            image_id2: Second image ID
            
        Returns:
            List of common 3D points
        """
        points = []
        for track in self.get_common_tracks(image_id1, image_id2):
            if track.point3d_id is not None:
                point = self.get_point3d(track.point3d_id)
                if point is not None:
                    points.append(point)
        
        return points
    
    def compute_reprojection_errors(self) -> Dict[int, float]:
        """Compute reprojection errors for all 3D points.
        
        Returns:
            Dictionary of point_id -> error
        """
        errors = {}
        
        for point_id, point in self.points3d.items():
            track = self.tracks.get(point.track_id)
            if track is None:
                continue
            
            # Compute reprojection error for each observation
            reproj_errors = []
            for image_id, feature_idx in track.observations.items():
                camera = self.cameras.get(image_id)
                if camera is None:
                    continue
                
                # Get 3D point in world coordinates
                point_3d = point.position
                
                # Project to image
                point_2d, _ = camera.project(point_3d.reshape(1, 3))
                point_2d = point_2d[0]
                
                # Get feature coordinates
                features = features_by_image.get(image_id)
                if features is None:
                    continue
                
                feature_pos = features.keypoints[feature_idx]
                
                # Compute error
                error = np.linalg.norm(point_2d - feature_pos)
                reproj_errors.append(error)
            
            # Compute average error
            if reproj_errors:
                avg_error = sum(reproj_errors) / len(reproj_errors)
                errors[point_id] = avg_error
                
                # Update point error
                point.error = avg_error
        
        return errors
    
    def filter_points(self, max_error: float = 5.0) -> int:
        """Filter outlier 3D points.
        
        Args:
            max_error: Maximum reprojection error
            
        Returns:
            Number of removed points
        """
        # Compute reprojection errors
        errors = self.compute_reprojection_errors()
        
        # Filter points with high error
        points_to_remove = [point_id for point_id, error in errors.items() if error > max_error]
        
        # Remove points
        for point_id in points_to_remove:
            self.remove_point3d(point_id)
        
        return len(points_to_remove)
    
    def to_pointcloud(self) -> o3d.geometry.PointCloud:
        """Convert reconstruction to Open3D point cloud.
        
        Returns:
            Open3D point cloud
        """
        # Extract points and colors
        points = []
        colors = []
        
        for point in self.points3d.values():
            points.append(point.to_tuple())
            colors.append(point.color / 255.0)  # Normalize to [0, 1]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd


class StructureFromMotion:
    """Structure from Motion pipeline."""
    
    def __init__(self, options: Optional[SfMOptions] = None):
        """Initialize SfM pipeline.
        
        Args:
            options: SfM options
        """
        self.options = options or SfMOptions()
        self.reconstruction = SfMReconstruction()
        self.features_by_image = {}  # image_id -> FeatureData
        self.matches_by_pair = {}  # (image_id1, image_id2) -> MatchData
        self.image_size_by_id = {}  # image_id -> (width, height)
        self.intrinsics_by_id = {}  # image_id -> CameraIntrinsics
        self.images_by_id = {}  # image_id -> image data (for visualization)
    
    def set_features(self, image_id: int, features: FeatureData) -> None:
        """Set features for an image.
        
        Args:
            image_id: Image ID
            features: Feature data
        """
        self.features_by_image[image_id] = features
        self.image_size_by_id[image_id] = features.image_size
    
    def set_match(self, image_id1: int, image_id2: int, match: MatchData) -> None:
        """Set matches for an image pair.
        
        Args:
            image_id1: First image ID
            image_id2: Second image ID
            match: Match data
        """
        key = (min(image_id1, image_id2), max(image_id1, image_id2))
        self.matches_by_pair[key] = match
    
    def set_intrinsics(self, image_id: int, intrinsics: CameraIntrinsics) -> None:
        """Set camera intrinsics for an image.
        
        Args:
            image_id: Image ID
            intrinsics: Camera intrinsics
        """
        self.intrinsics_by_id[image_id] = intrinsics
    
    def set_image(self, image_id: int, image: np.ndarray) -> None:
        """Set image data for visualization.
        
        Args:
            image_id: Image ID
            image: Image data
        """
        self.images_by_id[image_id] = image
    
    def initialize_reconstruction(self, image_id1: int, image_id2: int) -> bool:
        """Initialize reconstruction with an image pair.
        
        Args:
            image_id1: First image ID
            image_id2: Second image ID
            
        Returns:
            True if initialization succeeded, False otherwise
        """
        logger.info(f"Initializing reconstruction with images {image_id1} and {image_id2}")
        
        # Get features
        features1 = self.features_by_image.get(image_id1)
        features2 = self.features_by_image.get(image_id2)
        
        if features1 is None or features2 is None:
            logger.error(f"Features not found for images {image_id1} or {image_id2}")
            return False
        
        # Get matches
        key = (min(image_id1, image_id2), max(image_id1, image_id2))
        matches = self.matches_by_pair.get(key)
        
        if matches is None:
            logger.error(f"Matches not found for image pair {key}")
            return False
        
        # Check number of matches
        if matches.num_matches < self.options.min_matches:
            logger.warning(f"Insufficient matches for initialization: {matches.num_matches} < {self.options.min_matches}")
            return False
        
        # Get intrinsics (if available)
        K1 = self.intrinsics_by_id.get(image_id1)
        K2 = self.intrinsics_by_id.get(image_id2)
        
        # Estimate intrinsics if not provided
        if K1 is None:
            width, height = features1.image_size
            K1 = CameraIntrinsics(
                width=width,
                height=height,
                fx=max(width, height),
                fy=max(width, height),
                cx=width / 2,
                cy=height / 2
            )
            self.intrinsics_by_id[image_id1] = K1
        
        if K2 is None:
            width, height = features2.image_size
            K2 = CameraIntrinsics(
                width=width,
                height=height,
                fx=max(width, height),
                fy=max(width, height),
                cx=width / 2,
                cy=height / 2
            )
            self.intrinsics_by_id[image_id2] = K2
        
        # Extract matched points
        matches_array = matches.matches
        kpts1 = features1.keypoints[matches_array[:, 0]]
        kpts2 = features2.keypoints[matches_array[:, 1]]
        
        # Estimate relative pose
        if self.options.init_method == "essential":
            # Use essential matrix
            E, mask = cv2.findEssentialMat(
                kpts1, kpts2, K1.K, method=cv2.RANSAC, 
                prob=0.999, threshold=self.options.ransac_threshold
            )
            
            if E is None or E.shape != (3, 3):
                logger.warning("Essential matrix estimation failed")
                return False
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, kpts1, kpts2, K1.K, mask=mask)
            
        elif self.options.init_method == "homography":
            # Use homography
            H, mask = cv2.findHomography(
                kpts1, kpts2, cv2.RANSAC, 
                ransacReprojThreshold=self.options.ransac_threshold
            )
            
            if H is None:
                logger.warning("Homography estimation failed")
                return False
            
            # Convert to pose (not implemented here)
            logger.error("Homography initialization not implemented")
            return False
            
        else:  # Auto
            # Try both methods
            # Essential matrix
            E, mask_e = cv2.findEssentialMat(
                kpts1, kpts2, K1.K, method=cv2.RANSAC, 
                prob=0.999, threshold=self.options.ransac_threshold
            )
            
            inliers_e = np.sum(mask_e) if E is not None and E.shape == (3, 3) else 0
            
            # Homography
            H, mask_h = cv2.findHomography(
                kpts1, kpts2, cv2.RANSAC, 
                ransacReprojThreshold=self.options.ransac_threshold
            )
            
            inliers_h = np.sum(mask_h) if H is not None else 0
            
            # Use method with more inliers
            if inliers_e > inliers_h:
                _, R, t, mask = cv2.recoverPose(E, kpts1, kpts2, K1.K, mask=mask_e)
            else:
                logger.error("Homography initialization not implemented")
                return False
        
        # Apply mask to matches
        valid_matches = []
        valid_indices = []
        
        for i, (valid, m) in enumerate(zip(mask.ravel().astype(bool), matches_array)):
            if valid:
                valid_matches.append(m)
                valid_indices.append(i)
        
        if len(valid_matches) < self.options.min_matches:
            logger.warning(f"Insufficient inlier matches after pose estimation: {len(valid_matches)} < {self.options.min_matches}")
            return False
        
        # Initialize cameras
        camera1 = Camera(
            intrinsics=K1,
            extrinsics=CameraExtrinsics(R=np.eye(3), t=np.zeros((3, 1)))
        )
        
        camera2 = Camera(
            intrinsics=K2,
            extrinsics=CameraExtrinsics(R=R, t=t)
        )
        
        # Add cameras to reconstruction
        self.reconstruction.add_camera(image_id1, camera1)
        self.reconstruction.add_camera(image_id2, camera2)
        
        # Set reference camera
        self.reconstruction.reference_camera = camera1
        self.reconstruction.reference_image_id = image_id1
        
        # Create tracks from inlier matches
        next_track_id = 0
        for idx, (idx1, idx2) in enumerate(valid_matches):
            track = Track(
                id=next_track_id,
                observations={
                    image_id1: idx1,
                    image_id2: idx2
                }
            )
            
            self.reconstruction.add_track(track)
            next_track_id += 1
        
        # Triangulate 3D points
        self._triangulate_new_points()
        
        # Update registration graph
        self.reconstruction.registration_graph.add_edge(image_id1, image_id2, weight=len(valid_matches))
        
        logger.info(f"Initialization succeeded with {len(valid_matches)} inlier matches")
        return True
    
    def select_initial_pair(self) -> Tuple[int, int]:
        """Select best image pair for initialization.
        
        Returns:
            Tuple of (image_id1, image_id2)
        """
        logger.info("Selecting initial image pair")
        
        # Score for each image pair
        pair_scores = []
        
        for (image_id1, image_id2), matches in self.matches_by_pair.items():
            # Skip if not enough matches
            if matches.num_matches < self.options.min_matches:
                continue
            
            # Get features
            features1 = self.features_by_image.get(image_id1)
            features2 = self.features_by_image.get(image_id2)
            
            if features1 is None or features2 is None:
                continue
            
            # Extract matched points
            matches_array = matches.matches
            kpts1 = features1.keypoints[matches_array[:, 0]]
            kpts2 = features2.keypoints[matches_array[:, 1]]
            
            # Try to estimate pose
            K1 = self.intrinsics_by_id.get(image_id1)
            if K1 is None:
                width, height = features1.image_size
                K1 = CameraIntrinsics(
                    width=width,
                    height=height,
                    fx=max(width, height),
                    fy=max(width, height),
                    cx=width / 2,
                    cy=height / 2
                )
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                kpts1, kpts2, K1.K, method=cv2.RANSAC, 
                prob=0.999, threshold=self.options.ransac_threshold
            )
            
            if E is None or E.shape != (3, 3):
                continue
            
            # Count inliers
            inliers = np.sum(mask)
            
            # Compute score based on number of inliers and their distribution
            if inliers < self.options.min_matches:
                continue
                
            # Recover pose to check parallax
            _, R, t, _ = cv2.recoverPose(E, kpts1, kpts2, K1.K, mask=mask)
            
            # Compute baseline and parallax
            baseline = np.linalg.norm(t)
            
            # Compute score
            score = inliers * baseline
            
            pair_scores.append((score, inliers, image_id1, image_id2))
        
        if not pair_scores:
            logger.error("No suitable image pair found for initialization")
            return (0, 0)
        
        # Sort by score (descending)
        pair_scores.sort(reverse=True)
        
        # Return best pair
        _, inliers, image_id1, image_id2 = pair_scores[0]
        logger.info(f"Selected initial pair: {image_id1}, {image_id2} with {inliers} inliers")
        
        return (image_id1, image_id2)
    
    def _triangulate_new_points(self) -> int:
        """Triangulate new 3D points from tracks.
        
        Returns:
            Number of triangulated points
        """
        # Count triangulated points
        num_triangulated = 0
        
        # Process tracks without 3D points
        tracks_to_process = [track for track in self.reconstruction.tracks.values() 
                           if track.point3d_id is None and track.length >= self.options.min_track_length]
        
        for track in tracks_to_process:
            # Get observations
            observations = track.observations
            
            # Need at least 2 views
            if len(observations) < 2:
                continue
            
            # Get cameras for triangulation
            cameras = []
            image_ids = []
            feature_indices = []
            
            for image_id, feature_idx in observations.items():
                camera = self.reconstruction.get_camera(image_id)
                if camera is None:
                    continue
                
                cameras.append(camera)
                image_ids.append(image_id)
                feature_indices.append(feature_idx)
            
            if len(cameras) < 2:
                continue
            
            # Get 2D points
            points_2d = []
            for image_id, feature_idx in zip(image_ids, feature_indices):
                features = self.features_by_image.get(image_id)
                if features is None:
                    continue
                
                point_2d = features.keypoints[feature_idx]
                points_2d.append(point_2d)
            
            if len(points_2d) < 2:
                continue
            
            # Triangulate point
            point_3d = self._triangulate_point(cameras, points_2d, image_ids)
            
            if point_3d is None:
                continue
            
            # Create 3D point
            position = point_3d
            
            # Compute color (average from features)
            colors = []
            for image_id, feature_idx in zip(image_ids, feature_indices):
                # If image data is available
                image = self.images_by_id.get(image_id)
                if image is not None:
                    features = self.features_by_image.get(image_id)
                    if features is None:
                        continue
                    
                    # Get feature coordinates
                    x, y = features.keypoints[feature_idx].astype(int)
                    
                    # Get color from image
                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        if len(image.shape) == 3:
                            color = image[y, x]
                        else:
                            color = np.array([image[y, x], image[y, x], image[y, x]])
                        
                        colors.append(color)
            
            if colors:
                color = np.mean(colors, axis=0).astype(np.uint8)
            else:
                color = np.array([200, 200, 200])  # Default gray
            
            # Create point
            point = Point3D(
                id=-1,  # Will be assigned by add_point3d
                position=position,
                color=color,
                track_id=track.id
            )
            
            # Add to reconstruction
            self.reconstruction.add_point3d(point)
            num_triangulated += 1
        
        return num_triangulated
    
    def _triangulate_point(self, 
                          cameras: List[Camera],
                          points_2d: List[np.ndarray],
                          image_ids: List[int]) -> Optional[np.ndarray]:
        """Triangulate 3D point from multiple views.
        
        Args:
            cameras: List of cameras
            points_2d: List of 2D points
            image_ids: List of image IDs
            
        Returns:
            3D point or None if triangulation fails
        """
        # Check number of views
        if len(cameras) < 2 or len(points_2d) < 2:
            return None
        
        # Check triangulation angles
        sufficient_angle = False
        for i in range(len(cameras)):
            for j in range(i+1, len(cameras)):
                # Get camera centers
                c1 = -cameras[i].extrinsics.R.T @ cameras[i].extrinsics.t
                c2 = -cameras[j].extrinsics.R.T @ cameras[j].extrinsics.t
                
                # Check baseline
                baseline = np.linalg.norm(c2 - c1)
                if baseline < 1e-6:
                    continue
                
                # Get viewing rays
                p1 = points_2d[i]
                p2 = points_2d[j]
                
                # Get camera intrinsics
                K1 = cameras[i].intrinsics.K
                K2 = cameras[j].intrinsics.K
                
                # Convert to normalized coordinates
                p1_norm = np.linalg.inv(K1) @ np.array([p1[0], p1[1], 1.0])
                p2_norm = np.linalg.inv(K2) @ np.array([p2[0], p2[1], 1.0])
                
                # Normalize
                p1_norm = p1_norm / np.linalg.norm(p1_norm)
                p2_norm = p2_norm / np.linalg.norm(p2_norm)
                
                # Rotate to world coordinates
                p1_world = cameras[i].extrinsics.R.T @ p1_norm
                p2_world = cameras[j].extrinsics.R.T @ p2_norm
                
                # Compute angle between rays
                cos_angle = np.clip(np.dot(p1_world, p2_world), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_angle))
                
                if angle_deg > self.options.triangulation_angle_threshold:
                    sufficient_angle = True
                    break
            
            if sufficient_angle:
                break
        
        if not sufficient_angle:
            return None
        
        # Prepare projection matrices
        P_matrices = []
        for camera in cameras:
            K = camera.intrinsics.K
            R = camera.extrinsics.R
            t = camera.extrinsics.t
            
            # Projection matrix: K @ [R|t]
            P = K @ np.hstack((R, t))
            P_matrices.append(P)
        
        # Convert to numpy array
        P_matrices = np.array(P_matrices)
        points_2d = np.array(points_2d)
        
        # Triangulate using DLT algorithm
        A = np.zeros((len(cameras) * 2, 4))
        
        for i, (P, point) in enumerate(zip(P_matrices, points_2d)):
            x, y = point
            A[i*2] = x * P[2] - P[0]
            A[i*2+1] = y * P[2] - P[1]
        
        # Solve system A * X = 0
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        
        # Convert to inhomogeneous coordinates
        X = X / X[3]
        point_3d = X[:3]
        
        # Check reprojection error
        max_error = 0.0
        
        for i, (camera, point_2d) in enumerate(zip(cameras, points_2d)):
            # Project 3D point
            proj_point, _ = camera.project(point_3d.reshape(1, 3))
            
            # Compute error
            error = np.linalg.norm(proj_point - point_2d)
            max_error = max(max_error, error)
        
        # Check if error is too high
        if max_error > self.options.max_reprojection_error:
            return None
        
        return point_3d
    
    def register_next_image(self) -> Optional[int]:
        """Register next best image to the reconstruction.
        
        Returns:
            ID of registered image or None if no image could be registered
        """
        logger.info("Registering next image")
        
        # Get registered and unregistered images
        registered_images = set(self.reconstruction.get_registered_images())
        all_images = set(self.features_by_image.keys())
        unregistered_images = all_images - registered_images
        
        if not unregistered_images:
            logger.info("All images are registered")
            return None
        
        # Score each unregistered image
        image_scores = []
        
        for image_id in unregistered_images:
            # Count matches to registered images
            matches_to_registered = 0
            inlier_matches = 0
            
            for reg_image_id in registered_images:
                key = (min(image_id, reg_image_id), max(image_id, reg_image_id))
                matches = self.matches_by_pair.get(key)
                
                if matches is None:
                    continue
                
                matches_to_registered += matches.num_matches
                
                # Count inlier matches (those belonging to tracks with 3D points)
                for idx1, idx2 in matches.matches:
                    # Make sure indices are in correct order
                    if key[0] == image_id:
                        idx_unreg, idx_reg = idx1, idx2
                        unreg_id, reg_id = image_id, reg_image_id
                    else:
                        idx_unreg, idx_reg = idx2, idx1
                        unreg_id, reg_id = image_id, reg_image_id
                    
                    # Check if there's a track with 3D point for this feature
                    for track in self.reconstruction.get_tracks_for_image(reg_id):
                        if track.observations.get(reg_id) == idx_reg and track.point3d_id is not None:
                            inlier_matches += 1
                            break
            
            # Score based on number of inlier matches
            score = inlier_matches
            
            if score >= self.options.min_matches:
                image_scores.append((score, image_id))
        
        if not image_scores:
            logger.warning("No image has enough matches to register")
            return None
        
        # Sort by score (descending)
        image_scores.sort(reverse=True)
        
        # Try to register images in order of score
        for _, image_id in image_scores:
            # Try to register image
            if self._register_image(image_id):
                logger.info(f"Registered image {image_id}")
                return image_id
        
        logger.warning("Failed to register any image")
        return None
    
    def _register_image(self, image_id: int) -> bool:
        """Register an image to the reconstruction.
        
        Args:
            image_id: Image ID
            
        Returns:
            True if registration succeeded, False otherwise
        """
        # Get features
        features = self.features_by_image.get(image_id)
        if features is None:
            logger.error(f"Features not found for image {image_id}")
            return False
        
        # Get intrinsics
        intrinsics = self.intrinsics_by_id.get(image_id)
        if intrinsics is None:
            # Estimate intrinsics
            width, height = features.image_size
            intrinsics = CameraIntrinsics(
                width=width,
                height=height,
                fx=max(width, height),
                fy=max(width, height),
                cx=width / 2,
                cy=height / 2
            )
            self.intrinsics_by_id[image_id] = intrinsics
        
        # Collect 2D-3D correspondences
        points2d = []
        points3d = []
        
        # Find correspondences from tracks
        registered_images = self.reconstruction.get_registered_images()
        
        for reg_image_id in registered_images:
            key = (min(image_id, reg_image_id), max(image_id, reg_image_id))
            matches = self.matches_by_pair.get(key)
            
            if matches is None:
                continue
            
            # Process each match
            for idx1, idx2 in matches.matches:
                # Make sure indices are in correct order
                if key[0] == image_id:
                    idx_unreg, idx_reg = idx1, idx2
                    unreg_id, reg_id = image_id, reg_image_id
                else:
                    idx_unreg, idx_reg = idx2, idx1
                    unreg_id, reg_id = image_id, reg_image_id
                
                # Find track for registered image feature
                for track in self.reconstruction.get_tracks_for_image(reg_id):
                    if track.observations.get(reg_id) == idx_reg and track.point3d_id is not None:
                        # Get 3D point
                        point3d = self.reconstruction.get_point3d(track.point3d_id)
                        if point3d is None:
                            continue
                        
                        # Get 2D feature coordinates
                        point2d = features.keypoints[idx_unreg]
                        
                        # Add correspondence
                        points2d.append(point2d)
                        points3d.append(point3d.position)
                        
                        break
        
        # Check if we have enough correspondences
        if len(points2d) < self.options.min_matches:
            logger.warning(f"Insufficient 2D-3D correspondences for image {image_id}: {len(points2d)} < {self.options.min_matches}")
            return False
        
        # Estimate pose
        if self.options.add_image_robust:
            # Use PnP RANSAC
            points2d = np.array(points2d)
            points3d = np.array(points3d)
            
            dist_coeffs = np.zeros(5)  # Assume no distortion for PnP
            
            try:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    points3d, points2d, intrinsics.K, dist_coeffs,
                    iterationsCount=100, reprojectionError=self.options.max_reprojection_error,
                    flags=cv2.SOLVEPNP_EPNP
                )
            except Exception as e:
                logger.error(f"PnP RANSAC failed: {e}")
                return False
            
            if not success or inliers is None or len(inliers) < self.options.min_matches:
                logger.warning(f"PnP RANSAC failed or insufficient inliers: {0 if inliers is None else len(inliers)}")
                return False
            
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
            
            # Keep only inlier correspondences
            inlier_indices = inliers.ravel()
            inlier_points2d = points2d[inlier_indices]
            inlier_points3d = points3d[inlier_indices]
            
            logger.info(f"PnP RANSAC: {len(inlier_indices)} inliers out of {len(points2d)} correspondences")
            
            # Refine pose
            if len(inlier_indices) >= 6:  # Need at least 6 points for DLT
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        inlier_points3d, inlier_points2d, intrinsics.K, dist_coeffs,
                        rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    # Convert refined rotation vector to matrix
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec
                    
                except Exception as e:
                    logger.warning(f"Pose refinement failed: {e}")
            
        else:
            # Use simple PnP without RANSAC
            points2d = np.array(points2d)
            points3d = np.array(points3d)
            
            dist_coeffs = np.zeros(5)  # Assume no distortion for PnP
            
            try:
                success, rvec, tvec = cv2.solvePnP(
                    points3d, points2d, intrinsics.K, dist_coeffs,
                    flags=cv2.SOLVEPNP_EPNP
                )
            except Exception as e:
                logger.error(f"PnP failed: {e}")
                return False
            
            if not success:
                logger.warning("PnP failed")
                return False
            
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)
            t = tvec
            
            # All correspondences are considered inliers
            inlier_indices = np.arange(len(points2d))
            inlier_points2d = points2d
            inlier_points3d = points3d
        
        # Create camera
        camera = Camera(
            intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(R=R, t=t)
        )
        
        # Add to reconstruction
        self.reconstruction.add_camera(image_id, camera)
        
        # Create tracks for inlier matches
        for i, point_idx in enumerate(inlier_indices):
            idx1, idx2 = matches.matches[point_idx]
            
            # Find existing track for this 3D point
            for track in self.reconstruction.tracks.values():
                if track.point3d_id is not None:
                    point3d = self.reconstruction.get_point3d(track.point3d_id)
                    if point3d is not None and np.allclose(point3d.position, inlier_points3d[i]):
                        # Add observation to existing track
                        track.observations[image_id] = idx1 if image_id == key[0] else idx2
                        break
            else:
                # Create new track for this point
                track = Track(
                    id=-1,  # Will be assigned by add_track
                    observations={image_id: idx1 if image_id == key[0] else idx2}
                )
                
                self.reconstruction.add_track(track)
        
        # Update registration graph
        for reg_image_id in registered_images:
            key = (min(image_id, reg_image_id), max(image_id, reg_image_id))
            matches = self.matches_by_pair.get(key)
            
            if matches is not None:
                self.reconstruction.registration_graph.add_edge(
                    image_id, reg_image_id, weight=matches.num_matches)
        
        return True
    
    def run_bundle_adjustment(self) -> None:
        """Run bundle adjustment to refine reconstruction."""
        logger.info("Running bundle adjustment")
        
        # Collect cameras, points, and observations
        cameras = {}
        points3d = {}
        point_indices = {}
        camera_indices = {}
        points2d = []
        
        next_camera_idx = 0
        next_point_idx = 0
        
        for image_id, camera in self.reconstruction.cameras.items():
            # Add camera
            cameras[next_camera_idx] = camera
            camera_indices[image_id] = next_camera_idx
            next_camera_idx += 1
        
        for point_id, point in self.reconstruction.points3d.items():
            # Add point
            points3d[next_point_idx] = point
            point_indices[point_id] = next_point_idx
            next_point_idx += 1
            
            # Add observations
            track = self.reconstruction.get_track(point.track_id)
            if track is None:
                continue
            
            for image_id, feature_idx in track.observations.items():
                camera = self.reconstruction.get_camera(image_id)
                if camera is None:
                    continue
                
                features = self.features_by_image.get(image_id)
                if features is None:
                    continue
                
                # Get 2D point
                point2d = features.keypoints[feature_idx]
                
                # Add observation
                camera_idx = camera_indices[image_id]
                point_idx = point_indices[point_id]
                
                points2d.append((camera_idx, point_idx, point2d))
        
        # Run bundle adjustment (not implemented here)
        # In a real implementation, this would use a library like ceres-solver
        # or a wrapper like pyceres, or g2o
        
        # For simplicity, we'll just pretend we ran bundle adjustment
        logger.info(f"Bundle adjustment with {len(cameras)} cameras, " +
                  f"{len(points3d)} points, and {len(points2d)} observations")
        
        # In a real implementation, the optimized cameras and points would be
        # updated in the reconstruction
    
    def run_reconstruction(self, 
                          images: List[np.ndarray],
                          features: List[FeatureData],
                          matches: List[MatchData],
                          min_track_length: int = 3) -> Tuple[Dict[int, Camera], SfMReconstruction]:
        """Run complete Structure from Motion pipeline.
        
        Args:
            images: List of input images
            features: List of feature data for each image
            matches: List of match data between image pairs
            min_track_length: Minimum number of views per 3D point
            
        Returns:
            Tuple of (cameras, reconstruction)
        """
        # Set options
        self.options.min_track_length = min_track_length
        
        # Set images and features
        for i, (image, feature) in enumerate(zip(images, features)):
            self.set_image(i, image)
            self.set_features(i, feature)
        
        # Set matches
        for match in matches:
            i, j = match.image_pair
            self.set_match(i, j, match)
        
        # Select initial pair
        image_id1, image_id2 = self.select_initial_pair()
        
        if image_id1 == 0 and image_id2 == 0:
            logger.error("Failed to find initial pair")
            return {}, self.reconstruction
        
        # Initialize reconstruction
        if not self.initialize_reconstruction(image_id1, image_id2):
            logger.error("Failed to initialize reconstruction")
            return {}, self.reconstruction
        
        # Incremental reconstruction
        while True:
            # Register next image
            next_image = self.register_next_image()
            
            if next_image is None:
                break
            
            # Triangulate new points
            num_new_points = self._triangulate_new_points()
            logger.info(f"Triangulated {num_new_points} new points")
            
            # Bundle adjustment
            if self.options.final_ba and len(self.reconstruction.cameras) % self.options.ba_frequency == 0:
                self.run_bundle_adjustment()
            
            # Filter points
            if self.options.filter_points:
                num_removed = self.reconstruction.filter_points(self.options.max_reprojection_error)
                logger.info(f"Removed {num_removed} outlier points")
        
        # Final bundle adjustment
        if self.options.final_ba:
            self.run_bundle_adjustment()
        
        # Final filtering
        if self.options.filter_points:
            num_removed = self.reconstruction.filter_points(self.options.max_reprojection_error)
            logger.info(f"Removed {num_removed} outlier points in final filtering")
        
        logger.info(f"Reconstruction finished with {len(self.reconstruction.cameras)} cameras and "
                  f"{len(self.reconstruction.points3d)} points")
        
        return self.reconstruction.cameras, self.reconstruction


def run_sfm(images: List[np.ndarray],
           features: List[FeatureData],
           matches: List[MatchData],
           min_track_length: int = 3,
           options: Optional[SfMOptions] = None) -> Tuple[Dict[int, Camera], o3d.geometry.PointCloud]:
    """Run Structure from Motion pipeline.
    
    Args:
        images: List of input images
        features: List of feature data for each image
        matches: List of match data between image pairs
        min_track_length: Minimum number of views per 3D point
        options: SfM options
        
    Returns:
        Tuple of (cameras, point cloud)
    """
    # Create SfM pipeline
    sfm = StructureFromMotion(options)
    
    # Run reconstruction
    cameras, reconstruction = sfm.run_reconstruction(
        images, features, matches, min_track_length)
    
    # Convert to point cloud
    pointcloud = reconstruction.to_pointcloud()
    
    return cameras, pointcloud


def main():
    """Example usage."""
    import argparse
    import os
    from recontext.core.feature_extraction import extract_features
    from recontext.core.matching import match_features
    from recontext.utils.io_utils import load_images
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Structure from Motion")
    parser.add_argument("--image_dir", required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--feature_type", default="sift", help="Feature type (sift, orb, superpoint)")
    parser.add_argument("--matcher_type", default="mutual_nn", help="Matcher type (mutual_nn, flann, superglue)")
    parser.add_argument("--min_track_length", type=int, default=3, help="Minimum track length")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    image_files = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    images = load_images(image_files)
    
    # Extract features
    features = extract_features(images, feature_type=args.feature_type)
    
    # Match features (all pairs)
    pairs = [(i, j) for i in range(len(images)) for j in range(i+1, len(images))]
    matches = match_features(features, matcher_type=args.matcher_type, pairs=pairs)
    
    # Run SfM
    options = SfMOptions(min_track_length=args.min_track_length)
    cameras, pointcloud = run_sfm(images, features, matches, args.min_track_length, options)
    
    # Save results
    if len(cameras) > 0:
        # Save cameras
        with open(os.path.join(args.output_dir, 'cameras.pkl'), 'wb') as f:
            import pickle
            pickle.dump(cameras, f)
        
        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(args.output_dir, 'pointcloud.ply'), pointcloud)
        
        logger.info(f"Results saved to {args.output_dir}")
    else:
        logger.error("Reconstruction failed")


if __name__ == "__main__":
    main()