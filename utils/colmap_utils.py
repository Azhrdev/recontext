#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
COLMAP utility functions for structure from motion.

This module provides functions to run COLMAP as an external process 
and parse its output files into usable formats for the RECONTEXT pipeline.

Author: James Wei
Date: 2024-01-15
Last modified: 2024-03-10
"""

import os
import sys
import subprocess
import shutil
import logging
import numpy as np
import sqlite3
import struct
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from recontext.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from recontext.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

def is_colmap_installed() -> bool:
    """Check if COLMAP is installed and available in the system path.
    
    Returns:
        True if COLMAP is installed, False otherwise
    """
    try:
        subprocess.run(['colmap', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def find_colmap_binary() -> Optional[str]:
    """Find COLMAP binary in common installation locations.
    
    Returns:
        Path to COLMAP binary or None if not found
    """
    # Check if in PATH
    if is_colmap_installed():
        return 'colmap'
    
    # Try common installation locations
    common_locations = [
        '/usr/local/bin/colmap',
        '/usr/bin/colmap',
        'C:/Program Files/COLMAP/colmap.exe',
        os.path.expanduser('~/COLMAP/colmap.exe'),
    ]
    
    for location in common_locations:
        if os.path.isfile(location):
            return location
    
    # Not found
    return None

def run_colmap(image_dir: str, 
               output_dir: str, 
               quality: str = 'medium',
               gpu_index: int = 0,
               single_camera: bool = False,
               skip_calibration: bool = False,
               custom_options: Optional[Dict[str, str]] = None) -> bool:
    """Run COLMAP SfM pipeline.
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save COLMAP output
        quality: Reconstruction quality ('low', 'medium', 'high')
        gpu_index: GPU index to use (-1 for CPU)
        single_camera: Whether to use a single camera model for all images
        skip_calibration: Whether to skip camera calibration
        custom_options: Optional dictionary of custom COLMAP options
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure COLMAP is installed
    colmap_bin = find_colmap_binary()
    if colmap_bin is None:
        logger.error("COLMAP not found. Please install COLMAP and add it to PATH.")
        return False
    
    logger.info(f"Using COLMAP binary: {colmap_bin}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, 'database.db')
    sparse_dir = os.path.join(output_dir, 'sparse')
    ensure_dir(sparse_dir)
    
    # Quality settings
    quality_settings = {
        'low': {
            'SiftExtraction.max_num_features': '4096',
            'SiftExtraction.use_gpu': '1',
            'SiftMatching.guided_matching': '0',
            'Mapper.ba_global_max_num_iterations': '20',
            'Mapper.filter_max_reproj_error': '4.0',
        },
        'medium': {
            'SiftExtraction.max_num_features': '8192',
            'SiftExtraction.use_gpu': '1',
            'SiftMatching.guided_matching': '1',
            'Mapper.ba_global_max_num_iterations': '50',
            'Mapper.filter_max_reproj_error': '3.0',
        },
        'high': {
            'SiftExtraction.max_num_features': '16384',
            'SiftExtraction.use_gpu': '1',
            'SiftMatching.guided_matching': '1',
            'Mapper.ba_global_max_num_iterations': '100',
            'Mapper.filter_max_reproj_error': '2.0',
        }
    }
    
    # Use selected quality settings
    settings = quality_settings.get(quality.lower(), quality_settings['medium'])
    
    # Update with custom options if provided
    if custom_options:
        settings.update(custom_options)
    
    # Set GPU index
    settings['SiftExtraction.gpu_index'] = str(gpu_index)
    settings['SiftMatching.gpu_index'] = str(gpu_index)
    
    try:
        # Feature extraction
        logger.info("Running COLMAP feature extraction...")
        feature_extractor_args = [
            colmap_bin, 'feature_extractor',
            '--database_path', database_path,
            '--image_path', image_dir,
            '--ImageReader.single_camera', '1' if single_camera else '0',
        ]
        
        # Add settings to command
        for key, value in settings.items():
            if key.startswith('SiftExtraction.'):
                feature_extractor_args.extend(['--' + key, value])
        
        # Run feature extraction
        subprocess.run(feature_extractor_args, check=True)
        
        # Feature matching
        logger.info("Running COLMAP feature matching...")
        feature_matcher_args = [
            colmap_bin, 'exhaustive_matcher',
            '--database_path', database_path,
        ]
        
        # Add settings to command
        for key, value in settings.items():
            if key.startswith('SiftMatching.'):
                feature_matcher_args.extend(['--' + key, value])
        
        # Run feature matching
        subprocess.run(feature_matcher_args, check=True)
        
        # Sparse reconstruction
        logger.info("Running COLMAP mapper...")
        mapper_args = [
            colmap_bin, 'mapper',
            '--database_path', database_path,
            '--image_path', image_dir,
            '--output_path', sparse_dir,
        ]
        
        # Add settings to command
        for key, value in settings.items():
            if key.startswith('Mapper.'):
                mapper_args.extend(['--' + key, value])
        
        # Run mapper
        subprocess.run(mapper_args, check=True)
        
        # Convert to binary format
        model_path = os.path.join(sparse_dir, '0')
        if os.path.exists(model_path):
            logger.info("Converting COLMAP model to binary format...")
            subprocess.run([
                colmap_bin, 'model_converter',
                '--input_path', model_path,
                '--output_path', os.path.join(sparse_dir, 'model.bin'),
                '--output_type', 'binary'
            ], check=True)
        
        logger.info("COLMAP reconstruction completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP process failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running COLMAP: {e}")
        return False

def read_model_binary(path_to_model: str) -> Tuple[Dict, Dict, Dict]:
    """Read COLMAP binary model.
    
    Args:
        path_to_model: Path to COLMAP binary model file or directory
        
    Returns:
        Tuple of (cameras, images, points3D) dictionaries
    """
    if os.path.isdir(path_to_model):
        path_to_model = os.path.join(path_to_model, "model.bin")
    
    cameras, images, points3D = {}, {}, {}
    
    try:
        with open(path_to_model, "rb") as f:
            # Cameras
            num_cameras = read_next_bytes(f, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_id, model_id, width, height = read_next_bytes(f, 16, "IIII")
                
                num_params = read_next_bytes(f, 8, "Q")[0]
                params = read_next_bytes(f, num_params * 8, "d" * num_params)
                
                cameras[camera_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": params
                }
            
            # Images
            num_images = read_next_bytes(f, 8, "Q")[0]
            for _ in range(num_images):
                image_id, qw, qx, qy, qz, tx, ty, tz = read_next_bytes(f, 32, "idddddd")
                camera_id, name = read_next_bytes(f, 8, "ii")
                
                name_size = read_next_bytes(f, 8, "Q")[0]
                name_bytes = read_next_bytes(f, name_size, "c" * name_size)
                name = b"".join(name_bytes).decode("utf-8")
                
                num_points = read_next_bytes(f, 8, "Q")[0]
                point2D_ids = read_next_bytes(f, num_points * 4, "i" * num_points)
                point2D_xy = read_next_bytes(f, num_points * 2 * 8, "d" * (num_points * 2))
                point2D_xy = np.array(point2D_xy).reshape(num_points, 2)
                
                images[image_id] = {
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": camera_id,
                    "name": name,
                    "point2D_ids": np.array(point2D_ids),
                    "point2D_xy": point2D_xy
                }
            
            # Points3D
            num_points = read_next_bytes(f, 8, "Q")[0]
            for _ in range(num_points):
                point_id, x, y, z, r, g, b, error = read_next_bytes(f, 32, "idddBBBd")
                
                track_length = read_next_bytes(f, 8, "Q")[0]
                track_elements = read_next_bytes(f, track_length * 2 * 4, "ii" * track_length)
                track_elements = np.array(track_elements).reshape(track_length, 2)
                
                points3D[point_id] = {
                    "xyz": np.array([x, y, z]),
                    "rgb": np.array([r, g, b]),
                    "error": error,
                    "track": track_elements
                }
    except Exception as e:
        logger.error(f"Error reading COLMAP model: {e}")
        return {}, {}, {}
    
    return cameras, images, points3D

def read_next_bytes(f, num_bytes, format_char_sequence):
    """Read and unpack the next bytes from a binary file.
    
    Args:
        f: Binary file object
        num_bytes: Number of bytes to read
        format_char_sequence: Format string for struct.unpack
        
    Returns:
        Tuple of unpacked values
    """
    data = f.read(num_bytes)
    return struct.unpack(format_char_sequence, data)

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix.
    
    Args:
        qvec: Quaternion vector (w, x, y, z)
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = qvec
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R

def colmap_camera_to_intrinsics(camera: Dict) -> CameraIntrinsics:
    """Convert COLMAP camera parameters to CameraIntrinsics.
    
    Args:
        camera: COLMAP camera dictionary
        
    Returns:
        CameraIntrinsics object
    """
    model_id = camera["model_id"]
    width = camera["width"]
    height = camera["height"]
    params = camera["params"]
    
    # COLMAP camera models:
    # 0 = SIMPLE_PINHOLE (f, cx, cy)
    # 1 = PINHOLE (fx, fy, cx, cy)
    # 2 = SIMPLE_RADIAL (f, cx, cy, k1)
    # 3 = RADIAL (f, cx, cy, k1, k2)
    # 4 = OPENCV (fx, fy, cx, cy, k1, k2, p1, p2)
    # 5 = OPENCV_FISHEYE (fx, fy, cx, cy, k1, k2, k3, k4)
    # 6 = FULL_OPENCV (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
    # 7 = FOV (fx, fy, cx, cy, omega)
    # 8 = SIMPLE_RADIAL_FISHEYE (f, cx, cy, k1)
    # 9 = RADIAL_FISHEYE (f, cx, cy, k1, k2)
    
    if model_id == 0:  # SIMPLE_PINHOLE
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = k2 = p1 = p2 = k3 = 0.0
        model = "perspective"
    elif model_id == 1:  # PINHOLE
        fx, fy, cx, cy = params
        k1 = k2 = p1 = p2 = k3 = 0.0
        model = "perspective"
    elif model_id == 2:  # SIMPLE_RADIAL
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3]
        k2 = p1 = p2 = k3 = 0.0
        model = "perspective"
    elif model_id == 3:  # RADIAL
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1, k2 = params[3], params[4]
        p1 = p2 = k3 = 0.0
        model = "perspective"
    elif model_id == 4:  # OPENCV
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        k3 = 0.0
        model = "perspective"
    elif model_id == 5:  # OPENCV_FISHEYE
        fx, fy, cx, cy, k1, k2, k3, k4 = params
        p1 = p2 = 0.0
        model = "fisheye"
    else:
        # Default to pinhole model with no distortion
        logger.warning(f"Unsupported COLMAP camera model: {model_id}, defaulting to pinhole")
        fx = fy = params[0] if len(params) > 0 else 1000.0
        cx = params[1] if len(params) > 1 else width / 2
        cy = params[2] if len(params) > 2 else height / 2
        k1 = k2 = p1 = p2 = k3 = 0.0
        model = "perspective"
    
    return CameraIntrinsics(
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

def parse_colmap_output(colmap_dir: str) -> Tuple[Dict[int, Camera], Optional[Union[np.ndarray, 'o3d.geometry.PointCloud']]]:
    """Parse COLMAP output files and convert to RECONTEXT format.
    
    Args:
        colmap_dir: Directory containing COLMAP output
        
    Returns:
        Tuple of (cameras, pointcloud)
    """
    try:
        import open3d as o3d
        has_open3d = True
    except ImportError:
        has_open3d = False
        logger.warning("open3d not installed, pointcloud will be returned as numpy array")
    
    sparse_dir = os.path.join(colmap_dir, 'sparse')
    model_path = os.path.join(sparse_dir, 'model.bin')
    
    if not os.path.exists(model_path):
        # Try inside '0' directory
        model_path = os.path.join(sparse_dir, '0')
    
    if not os.path.exists(model_path):
        logger.error(f"COLMAP model not found in {colmap_dir}")
        return {}, None
    
    # Read COLMAP model
    colmap_cameras, colmap_images, colmap_points3D = read_model_binary(model_path)
    
    if not colmap_cameras or not colmap_images or not colmap_points3D:
        logger.error("Failed to read COLMAP model")
        return {}, None
    
    # Convert to RECONTEXT format
    cameras = {}
    
    for image_id, colmap_image in colmap_images.items():
        # Get camera intrinsics
        camera_id = colmap_image["camera_id"]
        colmap_camera = colmap_cameras.get(camera_id)
        
        if colmap_camera is None:
            logger.warning(f"Camera {camera_id} not found for image {image_id}")
            continue
        
        intrinsics = colmap_camera_to_intrinsics(colmap_camera)
        
        # Get camera extrinsics
        qvec = colmap_image["qvec"]
        tvec = colmap_image["tvec"]
        
        R = qvec2rotmat(qvec)
        t = tvec.reshape(3, 1)
        
        extrinsics = CameraExtrinsics(R=R, t=t)
        
        # Create camera
        camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics)
        
        # Add to cameras dictionary
        cameras[image_id] = camera
    
    # Extract 3D points
    points = []
    colors = []
    
    for point_id, colmap_point in colmap_points3D.items():
        points.append(colmap_point["xyz"])
        colors.append(colmap_point["rgb"])
    
    if not points:
        logger.warning("No 3D points found in COLMAP model")
        return cameras, None
    
    points = np.array(points)
    colors = np.array(colors) / 255.0  # Normalize to [0, 1]
    
    # Create pointcloud
    if has_open3d:
        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(points)
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        return cameras, pointcloud
    else:
        # Return as numpy arrays
        return cameras, points

def run_colmap_densify(colmap_dir: str, 
                      image_dir: str,
                      output_dir: str,
                      quality: str = 'medium',
                      gpu_index: int = 0) -> bool:
    """Run COLMAP dense reconstruction.
    
    Args:
        colmap_dir: Directory containing COLMAP sparse reconstruction
        image_dir: Directory containing input images
        output_dir: Directory to save dense reconstruction
        quality: Reconstruction quality ('low', 'medium', 'high')
        gpu_index: GPU index to use (-1 for CPU)
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure COLMAP is installed
    colmap_bin = find_colmap_binary()
    if colmap_bin is None:
        logger.error("COLMAP not found. Please install COLMAP and add it to PATH.")
        return False
    
    # Ensure directories exist
    sparse_dir = os.path.join(colmap_dir, 'sparse')
    if not os.path.exists(sparse_dir):
        logger.error(f"COLMAP sparse reconstruction not found in {sparse_dir}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Quality settings
    quality_settings = {
        'low': {
            'patch_match_stereo': {
                'max_image_size': '1000',
                'window_radius': '3',
                'window_step': '2',
                'num_iterations': '3',
                'gpu_index': str(gpu_index),
            },
            'stereo_fusion': {
                'min_num_pixels': '3',
                'max_reproj_error': '4.0',
                'max_depth_error': '0.1',
            }
        },
        'medium': {
            'patch_match_stereo': {
                'max_image_size': '1600',
                'window_radius': '5',
                'window_step': '1',
                'num_iterations': '5',
                'gpu_index': str(gpu_index),
            },
            'stereo_fusion': {
                'min_num_pixels': '5',
                'max_reproj_error': '2.0',
                'max_depth_error': '0.05',
            }
        },
        'high': {
            'patch_match_stereo': {
                'max_image_size': '2400',
                'window_radius': '7',
                'window_step': '1',
                'num_iterations': '7',
                'gpu_index': str(gpu_index),
            },
            'stereo_fusion': {
                'min_num_pixels': '7',
                'max_reproj_error': '1.0',
                'max_depth_error': '0.02',
            }
        }
    }
    
    # Use selected quality settings
    settings = quality_settings.get(quality.lower(), quality_settings['medium'])
    
    try:
        # Image undistortion
        logger.info("Running COLMAP image undistorter...")
        undistorter_args = [
            colmap_bin, 'image_undistorter',
            '--input_path', os.path.join(sparse_dir, '0'),
            '--image_path', image_dir,
            '--output_path', output_dir,
            '--output_type', 'COLMAP',
        ]
        
        # Run image undistorter
        subprocess.run(undistorter_args, check=True)
        
        # Patch match stereo
        logger.info("Running COLMAP patch match stereo...")
        stereo_args = [
            colmap_bin, 'patch_match_stereo',
            '--workspace_path', output_dir,
            '--workspace_format', 'COLMAP',
        ]
        
        # Add settings to command
        for key, value in settings['patch_match_stereo'].items():
            stereo_args.extend(['--' + key, value])
        
        # Run patch match stereo
        subprocess.run(stereo_args, check=True)
        
        # Stereo fusion
        logger.info("Running COLMAP stereo fusion...")
        fusion_args = [
            colmap_bin, 'stereo_fusion',
            '--workspace_path', output_dir,
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', os.path.join(output_dir, 'fused.ply'),
        ]
        
        # Add settings to command
        for key, value in settings['stereo_fusion'].items():
            fusion_args.extend(['--' + key, value])
        
        # Run stereo fusion
        subprocess.run(fusion_args, check=True)
        
        # Poisson surface reconstruction
        logger.info("Running COLMAP Poisson mesher...")
        mesher_args = [
            colmap_bin, 'poisson_mesher',
            '--input_path', os.path.join(output_dir, 'fused.ply'),
            '--output_path', os.path.join(output_dir, 'meshed-poisson.ply'),
            '--trim', '7',
        ]
        
        # Run Poisson mesher
        subprocess.run(mesher_args, check=True)
        
        # Delaunay surface reconstruction
        logger.info("Running COLMAP Delaunay mesher...")
        delaunay_args = [
            colmap_bin, 'delaunay_mesher',
            '--input_path', output_dir,
            '--output_path', os.path.join(output_dir, 'meshed-delaunay.ply'),
        ]
        
        # Run Delaunay mesher
        subprocess.run(delaunay_args, check=True)
        
        logger.info("COLMAP dense reconstruction completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP process failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running COLMAP: {e}")
        return False

def extract_features_from_database(database_path: str) -> Tuple[Dict, Dict]:
    """Extract features from COLMAP database.
    
    Args:
        database_path: Path to COLMAP database
        
    Returns:
        Tuple of (keypoints, descriptors) dictionaries
    """
    keypoints = {}
    descriptors = {}
    
    try:
        # Connect to the database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get keypoints
        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints;")
        for image_id, rows, cols, data in cursor:
            keypoints[image_id] = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        
        # Get descriptors
        cursor.execute("SELECT image_id, rows, cols, data FROM descriptors;")
        for image_id, rows, cols, data in cursor:
            descriptors[image_id] = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Error extracting features from database: {e}")
        return {}, {}
    
    return keypoints, descriptors

def extract_matches_from_database(database_path: str) -> Dict:
    """Extract matches from COLMAP database.
    
    Args:
        database_path: Path to COLMAP database
        
    Returns:
        Dictionary of matches
    """
    matches = defaultdict(list)
    
    try:
        # Connect to the database
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Get matches
        cursor.execute("SELECT pair_id, rows, cols, data FROM matches;")
        for pair_id, rows, cols, data in cursor:
            # Extract image IDs from pair_id
            image_id1 = pair_id % 2147483647
            image_id2 = pair_id // 2147483647
            
            # Parse match data
            match_data = np.frombuffer(data, dtype=np.uint32).reshape(rows, cols)
            
            # Add to matches dictionary
            matches[(image_id1, image_id2)] = match_data
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Error extracting matches from database: {e}")
        return {}
    
    return matches

def convert_colmap_to_open3d(colmap_dir: str) -> Optional['o3d.geometry.PointCloud']:
    """Convert COLMAP point cloud to Open3D format.
    
    Args:
        colmap_dir: Directory containing COLMAP output
        
    Returns:
        Open3D point cloud or None if failed
    """
    """Convert COLMAP point cloud to Open3D format.
    
    Args:
        colmap_dir: Directory containing COLMAP output
        
    Returns:
        Open3D point cloud or None if failed
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("open3d not installed, cannot convert point cloud")
        return None
    
    # Try to find dense reconstruction first
    ply_path = os.path.join(colmap_dir, 'fused.ply')
    
    if not os.path.exists(ply_path):
        # Try sparse reconstruction
        sparse_dir = os.path.join(colmap_dir, 'sparse')
        
        # Try model.ply
        ply_path = os.path.join(sparse_dir, 'model.ply')
        
        if not os.path.exists(ply_path):
            # Try to convert from binary model
            model_path = os.path.join(sparse_dir, 'model.bin')
            
            if not os.path.exists(model_path):
                # Try '0' directory
                model_path = os.path.join(sparse_dir, '0')
                
                if not os.path.exists(model_path):
                    logger.error(f"No COLMAP point cloud found in {colmap_dir}")
                    return None
            
            # Read binary model
            cameras, images, points3D = read_model_binary(model_path)
            
            if not points3D:
                logger.error("No 3D points found in COLMAP model")
                return None
            
            # Create point cloud
            points = []
            colors = []
            
            for point_id, colmap_point in points3D.items():
                points.append(colmap_point["xyz"])
                colors.append(colmap_point["rgb"])
            
            if not points:
                logger.error("No 3D points found in COLMAP model")
                return None
            
            points = np.array(points)
            colors = np.array(colors) / 255.0  # Normalize to [0, 1]
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd
            
    # Try to load PLY file
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        logger.info(f"Loaded point cloud from {ply_path}")
        return pcd
    except Exception as e:
        logger.error(f"Failed to load point cloud: {e}")
        return None