#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset handling utilities for managing and processing 3D reconstruction datasets.

This module provides classes and functions for loading, processing, and managing 
various 3D reconstruction datasets in consistent formats for the RECONTEXT pipeline.

Author: Michael Chen
Date: 2024-01-15
Last modified: 2024-03-12
"""

import os
import json
import numpy as np
import cv2
import logging
import glob
import yaml
import zipfile
import tarfile
import shutil
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from tqdm import tqdm

from recontext.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from recontext.utils.io_utils import ensure_dir, download_file, load_images, is_valid_image
from recontext.utils.colmap_utils import parse_colmap_output
from recontext.config.paths import get_data_dir

logger = logging.getLogger(__name__)

class DatasetType(Enum):
    """Enumeration of supported dataset types."""
    IMAGES_ONLY = "images_only"  # Just a collection of images
    COLMAP = "colmap"            # COLMAP format with sparse/dense reconstruction
    SCANNET = "scannet"          # ScanNet dataset
    ETH3D = "eth3d"              # ETH3D dataset
    HYPERSIM = "hypersim"        # Hypersim dataset
    CUSTOM = "custom"            # Custom format with specific structure


@dataclass
class DatasetInfo:
    """Dataset information."""
    name: str                               # Dataset name
    type: DatasetType                       # Dataset type
    path: str                               # Path to dataset
    num_images: int = 0                     # Number of images
    has_reconstruction: bool = False        # Whether dataset includes reconstruction
    has_gt_geometry: bool = False          # Whether dataset includes ground truth geometry
    has_gt_semantics: bool = False         # Whether dataset includes ground truth semantics
    metadata: Dict[str, Any] = None         # Additional metadata
    license: str = "Unknown"                # Dataset license information


class Dataset:
    """Base class for datasets."""
    
    def __init__(self, path: str, name: Optional[str] = None, load_images: bool = False):
        """Initialize dataset.
        
        Args:
            path: Path to dataset
            name: Optional dataset name (defaults to directory name)
            load_images: Whether to load images into memory
        """
        self.path = os.path.abspath(path)
        self.name = name or os.path.basename(self.path)
        self.load_images_in_memory = load_images
        
        # Initialize empty containers
        self.images = {}  # image_id -> image data (if load_images is True)
        self.image_paths = {}  # image_id -> image path
        self.cameras = {}  # image_id -> Camera
        self.intrinsics = {}  # intrinsics_id -> CameraIntrinsics
        self.extrinsics = {}  # image_id -> CameraExtrinsics
        self.semantics = {}  # image_id -> semantic labels
        self.metadata = {}  # Additional dataset metadata
        
        # Dataset info
        self.info = DatasetInfo(
            name=self.name,
            type=DatasetType.CUSTOM,
            path=self.path
        )
        
        # Initialization status
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize dataset by loading metadata and paths.
        
        Returns:
            Success status
        """
        raise NotImplementedError("Subclasses must implement initialize")
    
    def get_image(self, image_id: int) -> np.ndarray:
        """Get image by ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Image data
        """
        if not self.initialized:
            self.initialize()
        
        # If images are loaded in memory, return directly
        if self.load_images_in_memory and image_id in self.images:
            return self.images[image_id]
        
        # Otherwise, load from disk
        if image_id in self.image_paths:
            image_path = self.image_paths[image_id]
            image = cv2.imread(image_path)
            if image is not None:
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
            else:
                logger.error(f"Failed to load image {image_id} from {image_path}")
                return None
        else:
            logger.error(f"Image {image_id} not found in dataset")
            return None
    
    def get_camera(self, image_id: int) -> Optional[Camera]:
        """Get camera by image ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Camera object or None if not found
        """
        if not self.initialized:
            self.initialize()
        
        return self.cameras.get(image_id)
    
    def get_intrinsics(self, intrinsics_id: int) -> Optional[CameraIntrinsics]:
        """Get camera intrinsics by ID.
        
        Args:
            intrinsics_id: Intrinsics ID
            
        Returns:
            CameraIntrinsics object or None if not found
        """
        if not self.initialized:
            self.initialize()
        
        return self.intrinsics.get(intrinsics_id)
    
    def get_extrinsics(self, image_id: int) -> Optional[CameraExtrinsics]:
        """Get camera extrinsics by image ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            CameraExtrinsics object or None if not found
        """
        if not self.initialized:
            self.initialize()
        
        return self.extrinsics.get(image_id)
    
    def get_semantics(self, image_id: int) -> Optional[np.ndarray]:
        """Get semantic labels by image ID.
        
        Args:
            image_id: Image ID
            
        Returns:
            Semantic labels or None if not found
        """
        if not self.initialized:
            self.initialize()
        
        return self.semantics.get(image_id)
    
    def get_all_images(self) -> Dict[int, np.ndarray]:
        """Get all images.
        
        Returns:
            Dictionary of image_id -> image data
        """
        if not self.initialized:
            self.initialize()
        
        result = {}
        
        # If images are loaded in memory, return directly
        if self.load_images_in_memory:
            return self.images
        
        # Otherwise, load from disk
        for image_id, image_path in tqdm(self.image_paths.items(), desc="Loading images"):
            image = cv2.imread(image_path)
            if image is not None:
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result[image_id] = image
        
        return result
    
    def get_all_cameras(self) -> Dict[int, Camera]:
        """Get all cameras.
        
        Returns:
            Dictionary of image_id -> Camera
        """
        if not self.initialized:
            self.initialize()
        
        return self.cameras
    
    def get_all_intrinsics(self) -> Dict[int, CameraIntrinsics]:
        """Get all camera intrinsics.
        
        Returns:
            Dictionary of intrinsics_id -> CameraIntrinsics
        """
        if not self.initialized:
            self.initialize()
        
        return self.intrinsics
    
    def get_all_extrinsics(self) -> Dict[int, CameraExtrinsics]:
        """Get all camera extrinsics.
        
        Returns:
            Dictionary of image_id -> CameraExtrinsics
        """
        if not self.initialized:
            self.initialize()
        
        return self.extrinsics
    
    def get_all_semantics(self) -> Dict[int, np.ndarray]:
        """Get all semantic labels.
        
        Returns:
            Dictionary of image_id -> semantic labels
        """
        if not self.initialized:
            self.initialize()
        
        return self.semantics
    
    def get_info(self) -> DatasetInfo:
        """Get dataset information.
        
        Returns:
            Dataset information
        """
        if not self.initialized:
            self.initialize()
        
        return self.info
    
    def save_metadata(self, output_path: Optional[str] = None) -> str:
        """Save dataset metadata to file.
        
        Args:
            output_path: Optional output path (defaults to dataset_path/metadata.json)
            
        Returns:
            Path to saved metadata file
        """
        if not self.initialized:
            self.initialize()
        
        # Default output path
        if output_path is None:
            output_path = os.path.join(self.path, "metadata.json")
        
        # Ensure directory exists
        ensure_dir(os.path.dirname(output_path))
        
        # Create metadata
        metadata = {
            "name": self.name,
            "type": self.info.type.value,
            "num_images": len(self.image_paths),
            "has_reconstruction": bool(self.cameras),
            "has_gt_geometry": self.info.has_gt_geometry,
            "has_gt_semantics": bool(self.semantics),
            "license": self.info.license,
            "metadata": self.metadata
        }
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {output_path}")
        
        return output_path


class ImagesDataset(Dataset):
    """Dataset consisting of a directory of images."""
    
    def __init__(self, path: str, name: Optional[str] = None, load_images: bool = False):
        """Initialize images dataset.
        
        Args:
            path: Path to dataset (directory containing images)
            name: Optional dataset name (defaults to directory name)
            load_images: Whether to load images into memory
        """
        super().__init__(path, name, load_images)
        self.info.type = DatasetType.IMAGES_ONLY
    
    def initialize(self) -> bool:
        """Initialize dataset by discovering images.
        
        Returns:
            Success status
        """
        # Check if path exists
        if not os.path.exists(self.path):
            logger.error(f"Dataset path does not exist: {self.path}")
            return False
        
        # Find image files
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            image_files.extend(glob.glob(os.path.join(self.path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(self.path, f"*{ext.upper()}")))
        
        # Sort for consistent ordering
        image_files.sort()
        
        if not image_files:
            logger.error(f"No images found in {self.path}")
            return False
        
        # Add to dataset
        for i, image_file in enumerate(image_files):
            # Check if valid image
            if is_valid_image(image_file):
                self.image_paths[i] = image_file
                
                # Load image if requested
                if self.load_images_in_memory:
                    image = cv2.imread(image_file)
                    if image is not None:
                        # Convert to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.images[i] = image
        
        # Update dataset info
        self.info.num_images = len(self.image_paths)
        
        logger.info(f"Initialized dataset with {self.info.num_images} images")
        
        self.initialized = True
        return True


class ColmapDataset(Dataset):
    """Dataset from COLMAP reconstruction."""
    
    def __init__(self, path: str, name: Optional[str] = None, load_images: bool = False):
        """Initialize COLMAP dataset.
        
        Args:
            path: Path to dataset (directory containing COLMAP reconstruction)
            name: Optional dataset name (defaults to directory name)
            load_images: Whether to load images into memory
        """
        super().__init__(path, name, load_images)
        self.info.type = DatasetType.COLMAP
        self.colmap_path = None
        self.images_path = None
    
    def initialize(self) -> bool:
        """Initialize dataset by loading COLMAP reconstruction.
        
        Returns:
            Success status
        """
        # Check if path exists
        if not os.path.exists(self.path):
            logger.error(f"Dataset path does not exist: {self.path}")
            return False
        
        # Find COLMAP reconstruction
        colmap_candidates = [
            os.path.join(self.path, "colmap"),
            os.path.join(self.path, "sparse"),
            self.path
        ]
        
        for candidate in colmap_candidates:
            if os.path.exists(os.path.join(candidate, "sparse")) or \
               os.path.exists(os.path.join(candidate, "dense")):
                self.colmap_path = candidate
                break
        
        if self.colmap_path is None:
            logger.error(f"No COLMAP reconstruction found in {self.path}")
            return False
        
        # Find images directory
        image_candidates = [
            os.path.join(self.path, "images"),
            os.path.join(self.path, "imgs"),
            os.path.join(self.colmap_path, "images"),
            os.path.dirname(self.colmap_path)
        ]
        
        for candidate in image_candidates:
            # Check if directory exists and contains images
            if os.path.isdir(candidate):
                image_files = []
                for ext in [".jpg", ".jpeg", ".png"]:
                    image_files.extend(glob.glob(os.path.join(candidate, f"*{ext}")))
                
                if image_files:
                    self.images_path = candidate
                    break
        
        if self.images_path is None:
            logger.warning(f"No images directory found for COLMAP reconstruction in {self.path}")
            # Can still continue without images
        
        # Load COLMAP reconstruction
        logger.info(f"Loading COLMAP reconstruction from {self.colmap_path}")
        colmap_cameras, pointcloud = parse_colmap_output(self.colmap_path)
        
        if not colmap_cameras:
            logger.error(f"Failed to load COLMAP reconstruction from {self.colmap_path}")
            return False
        
        # Add cameras to dataset
        self.cameras = colmap_cameras
        
        # Extract intrinsics and extrinsics
        for image_id, camera in colmap_cameras.items():
            self.intrinsics[image_id] = camera.intrinsics
            self.extrinsics[image_id] = camera.extrinsics
        
        # Find image files if images_path is available
        if self.images_path is not None:
            # COLMAP image naming convention: {image_id}.jpg or {image_name}.jpg
            # Try to match by both ID and name
            image_files = {}
            for ext in [".jpg", ".jpeg", ".png"]:
                for image_file in glob.glob(os.path.join(self.images_path, f"*{ext}")):
                    # Try to extract image ID from filename
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    
                    # Check if filename is an integer (image ID)
                    if base_name.isdigit():
                        image_id = int(base_name)
                        image_files[image_id] = image_file
                    else:
                        # Try to match by image name in COLMAP
                        for camera_id, camera in colmap_cameras.items():
                            # Extract image name from COLMAP if available
                            image_name = str(camera_id)  # Default to camera ID
                            # TODO: Extract actual image name from COLMAP if needed
                            
                            if base_name == image_name:
                                image_files[camera_id] = image_file
                                break
            
            # Add to dataset
            for image_id, image_file in image_files.items():
                self.image_paths[image_id] = image_file
                
                # Load image if requested
                if self.load_images_in_memory:
                    image = cv2.imread(image_file)
                    if image is not None:
                        # Convert to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.images[image_id] = image
        
        # Update dataset info
        self.info.num_images = len(self.cameras)
        self.info.has_reconstruction = True
        
        logger.info(f"Initialized COLMAP dataset with {self.info.num_images} cameras")
        
        self.initialized = True
        return True


class ScanNetDataset(Dataset):
    """ScanNet dataset.
    
    The ScanNet dataset is an RGB-D video dataset containing 2.5 million views in more than 1500 scans,
    annotated with 3D camera poses, surface reconstructions, and semantic segmentations.
    http://www.scan-net.org/
    """
    
    def __init__(self, path: str, name: Optional[str] = None, load_images: bool = False):
        """Initialize ScanNet dataset.
        
        Args:
            path: Path to dataset (ScanNet scene directory)
            name: Optional dataset name (defaults to directory name)
            load_images: Whether to load images into memory
        """
        super().__init__(path, name, load_images)
        self.info.type = DatasetType.SCANNET
        self.info.has_gt_geometry = True
        self.info.has_gt_semantics = True
        self.info.license = "ScanNet Academic License"
    
    def initialize(self) -> bool:
        """Initialize dataset by loading ScanNet metadata and poses.
        
        Returns:
            Success status
        """
        # Check if path exists
        if not os.path.exists(self.path):
            logger.error(f"Dataset path does not exist: {self.path}")
            return False
        
        # Extract scene ID from path
        scene_id = os.path.basename(self.path)
        if not scene_id.startswith("scene"):
            scene_id = os.path.basename(os.path.dirname(self.path))
        
        self.metadata["scene_id"] = scene_id
        
        # Find pose file
        pose_file = os.path.join(self.path, f"{scene_id}.txt")
        if not os.path.exists(pose_file):
            # Try alternate location
            pose_file = os.path.join(self.path, "pose", f"{scene_id}.txt")
            
            if not os.path.exists(pose_file):
                logger.error(f"Pose file not found for scene {scene_id}")
                return False
        
        # Find color images
        color_dir = os.path.join(self.path, "color")
        if not os.path.exists(color_dir):
            logger.error(f"Color image directory not found for scene {scene_id}")
            return False
        
        # Find semantic labels
        label_dir = os.path.join(self.path, "label")
        has_semantics = os.path.exists(label_dir)
        
        # Parse pose file
        # ScanNet pose format: each line contains 4x4 camera-to-world matrix
        poses = []
        try:
            with open(pose_file, "r") as f:
                lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse 4x4 matrix
                    values = [float(x) for x in line.split()]
                    if len(values) != 16:
                        continue
                    
                    matrix = np.array(values).reshape(4, 4)
                    poses.append(matrix)
        except Exception as e:
            logger.error(f"Failed to parse pose file {pose_file}: {e}")
            return False
        
        # Find color images
        color_files = sorted(glob.glob(os.path.join(color_dir, "*.jpg")))
        if not color_files:
            color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
            
        if not color_files:
            logger.error(f"No color images found in {color_dir}")
            return False
        
        # ScanNet intrinsics (default values for ScanNet)
        # Can be overridden by reading sensor file if available
        width, height = 640, 480
        fx, fy = 577.870, 577.870
        cx, cy = 319.5, 239.5
        
        # Try to load intrinsics from sensor file
        sensor_file = os.path.join(self.path, f"{scene_id}.txt.sensor")
        if os.path.exists(sensor_file):
            try:
                with open(sensor_file, "r") as f:
                    for line in f:
                        if line.startswith("m_calibrationColorIntrinsic"):
                            values = [float(x) for x in line.split("=")[1].split()]
                            if len(values) == 16:
                                # Extract intrinsics from 4x4 matrix
                                intrinsic_matrix = np.array(values).reshape(4, 4)
                                fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
                                cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
                                break
            except Exception as e:
                logger.warning(f"Failed to parse sensor file {sensor_file}: {e}")
                # Continue with default values
        
        # Create intrinsics
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy
        )
        
        self.intrinsics[0] = intrinsics
        
        # Process each image
        for i, color_file in enumerate(color_files):
            # Get corresponding pose
            if i < len(poses):
                pose = poses[i]
                
                # ScanNet poses are camera-to-world, but we need world-to-camera
                # Invert the pose
                pose_inv = np.linalg.inv(pose)
                
                # Extract rotation and translation
                R = pose_inv[:3, :3]
                t = pose_inv[:3, 3].reshape(3, 1)
                
                # Create camera
                extrinsics = CameraExtrinsics(R=R, t=t)
                camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics)
                
                self.cameras[i] = camera
                self.extrinsics[i] = extrinsics
            
            # Add image path
            self.image_paths[i] = color_file
            
            # Load image if requested
            if self.load_images_in_memory:
                image = cv2.imread(color_file)
                if image is not None:
                    # Convert to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.images[i] = image
            
            # Load semantic label if available
            if has_semantics:
                # Get corresponding label file
                label_file = os.path.join(label_dir, os.path.basename(color_file).replace(".jpg", ".png").replace(".png", ".png"))
                
                if os.path.exists(label_file):
                    label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
                    if label is not None:
                        self.semantics[i] = label
        
        # Update dataset info
        self.info.num_images = len(self.image_paths)
        self.info.has_reconstruction = True
        self.info.has_gt_semantics = has_semantics
        
        logger.info(f"Initialized ScanNet dataset with {self.info.num_images} images")
        
        self.initialized = True
        return True


class ETH3DDataset(Dataset):
    """ETH3D dataset.
    
    The ETH3D dataset contains high-resolution multi-view stereo data with ground truth.
    https://www.eth3d.net/
    """
    
    def __init__(self, path: str, name: Optional[str] = None, load_images: bool = False):
        """Initialize ETH3D dataset.
        
        Args:
            path: Path to dataset (ETH3D scene directory)
            name: Optional dataset name (defaults to directory name)
            load_images: Whether to load images into memory
        """
        super().__init__(path, name, load_images)
        self.info.type = DatasetType.ETH3D
        self.info.has_gt_geometry = True
        self.info.license = "ETH3D Terms of Use"
    
    def initialize(self) -> bool:
        """Initialize dataset by loading ETH3D metadata and poses.
        
        Returns:
            Success status
        """
        # Check if path exists
        if not os.path.exists(self.path):
            logger.error(f"Dataset path does not exist: {self.path}")
            return False
        
        # Try to find COLMAP reconstruction
        colmap_dir = os.path.join(self.path, "dslr_calibration_undistorted")
        
        if not os.path.exists(colmap_dir):
            # Try alternate location
            colmap_dir = os.path.join(self.path, "dslr_calibration")
            
            if not os.path.exists(colmap_dir):
                logger.error(f"COLMAP reconstruction not found in {self.path}")
                return False
        
        # Load COLMAP reconstruction
        logger.info(f"Loading COLMAP reconstruction from {colmap_dir}")
        colmap_cameras, pointcloud = parse_colmap_output(colmap_dir)
        
        if not colmap_cameras:
            logger.error(f"Failed to load COLMAP reconstruction from {colmap_dir}")
            return False
        
        # Add cameras to dataset
        self.cameras = colmap_cameras
        
        # Extract intrinsics and extrinsics
        for image_id, camera in colmap_cameras.items():
            self.intrinsics[image_id] = camera.intrinsics
            self.extrinsics[image_id] = camera.extrinsics
        
        # Find images directory
        images_dir = os.path.join(self.path, "images")
        if not os.path.exists(images_dir):
            # Try alternate location
            images_dir = os.path.join(self.path, "dslr_images_undistorted")
            
            if not os.path.exists(images_dir):
                images_dir = os.path.join(self.path, "dslr_images")
                
                if not os.path.exists(images_dir):
                    logger.warning(f"Images directory not found in {self.path}")
                    # Can still continue without images
                    images_dir = None
        
        # Find image files if images_dir is available
        if images_dir is not None:
            # ETH3D image naming convention varies, try to match by name
            image_files = {}
            for ext in [".jpg", ".jpeg", ".png"]:
                for image_file in glob.glob(os.path.join(images_dir, f"*{ext}")):
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    
                    # Try to match by image name in COLMAP
                    matched = False
                    for camera_id, camera in colmap_cameras.items():
                        # Extract image name from COLMAP if available
                        image_name = str(camera_id)  # Default to camera ID
                        # TODO: Extract actual image name from COLMAP if needed
                        
                        if base_name == image_name:
                            image_files[camera_id] = image_file
                            matched = True
                            break
                    
                    # If not matched by name, assign sequential ID
                    if not matched:
                        image_files[len(image_files)] = image_file
            
            # Add to dataset
            for image_id, image_file in image_files.items():
                self.image_paths[image_id] = image_file
                
                # Load image if requested
                if self.load_images_in_memory:
                    image = cv2.imread(image_file)
                    if image is not None:
                        # Convert to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        self.images[image_id] = image
        
        # Update dataset info
        self.info.num_images = len(self.cameras)
        self.info.has_reconstruction = True
        
        logger.info(f"Initialized ETH3D dataset with {self.info.num_images} cameras")
        
        self.initialized = True
        return True


class HypersimDataset(Dataset):
    """Hypersim dataset.
    
    The Hypersim dataset is a photorealistic synthetic dataset for indoor scene understanding.
    https://github.com/apple/ml-hypersim
    """
    
    def __init__(self, path: str, name: Optional[str] = None, load_images: bool = False):
        """Initialize Hypersim dataset.
        
        Args:
            path: Path to dataset (Hypersim scene directory)
            name: Optional dataset name (defaults to directory name)
            load_images: Whether to load images into memory
        """
        super().__init__(path, name, load_images)
        self.info.type = DatasetType.HYPERSIM
        self.info.has_gt_geometry = True
        self.info.has_gt_semantics = True
        self.info.license = "Hypersim Dataset License"
    
    def initialize(self) -> bool:
        """Initialize dataset by loading Hypersim metadata and poses.
        
        Returns:
            Success status
        """
        # Check if path exists
        if not os.path.exists(self.path):
            logger.error(f"Dataset path does not exist: {self.path}")
            return False
        
        # Get scene ID from path
        scene_id = os.path.basename(self.path)
        self.metadata["scene_id"] = scene_id
        
        # Find camera parameter file
        camera_params_file = os.path.join(self.path, "_detail", "cam_params.json")
        if not os.path.exists(camera_params_file):
            logger.error(f"Camera parameter file not found in {self.path}")
            return False
        
        # Load camera parameters
        try:
            with open(camera_params_file, "r") as f:
                camera_params = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load camera parameters: {e}")
            return False
        
        # Find RGB images
        images_dir = os.path.join(self.path, "images", "scene")
        if not os.path.exists(images_dir):
            logger.error(f"RGB image directory not found in {self.path}")
            return False
        
        # Find semantic labels
        semantics_dir = os.path.join(self.path, "images", "scene_semantics")
        has_semantics = os.path.exists(semantics_dir)
        
        # Process each camera view
        for camera_name, params in camera_params.items():
            # Extract camera ID
            try:
                camera_id = int(camera_name.split("_")[-1])
            except ValueError:
                camera_id = len(self.cameras)
            
            # Parse camera parameters
            # Hypersim uses OpenGL convention (different from OpenCV)
            try:
                intrinsic_matrix = np.array(params["intrinsic_mat"])
                extrinsic_matrix = np.array(params["extrinsic_mat"])
                
                # Extract intrinsics
                width = params["width"]
                height = params["height"]
                fx = intrinsic_matrix[0, 0]
                fy = intrinsic_matrix[1, 1]
                cx = intrinsic_matrix[0, 2]
                cy = intrinsic_matrix[1, 2]
                
                # Create intrinsics
                intrinsics = CameraIntrinsics(
                    width=width,
                    height=height,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy
                )
                
                # Extract extrinsics (convert from OpenGL to OpenCV)
                # OpenGL: Y-up, Z-backward
                # OpenCV: Y-down, Z-forward
                # Need to flip Y and Z axes
                R_gl = extrinsic_matrix[:3, :3]
                t_gl = extrinsic_matrix[:3, 3].reshape(3, 1)
                
                # Convert to OpenCV convention
                R_cv = R_gl.copy()
                R_cv[1, :] *= -1  # Flip Y
                R_cv[2, :] *= -1  # Flip Z
                
                t_cv = t_gl.copy()
                t_cv[1] *= -1  # Flip Y
                t_cv[2] *= -1  # Flip Z
                
                # Create extrinsics
                extrinsics = CameraExtrinsics(R=R_cv, t=t_cv)
                
                # Create camera
                camera = Camera(intrinsics=intrinsics, extrinsics=extrinsics)
                
                # Add to dataset
                self.cameras[camera_id] = camera
                self.intrinsics[camera_id] = intrinsics
                self.extrinsics[camera_id] = extrinsics
                
                # Find RGB image file
                rgb_files = glob.glob(os.path.join(images_dir, f"{camera_name}_*.jpg"))
                if not rgb_files:
                    rgb_files = glob.glob(os.path.join(images_dir, f"{camera_name}_*.png"))
                
                if rgb_files:
                    rgb_file = sorted(rgb_files)[0]  # Take first frame if multiple
                    self.image_paths[camera_id] = rgb_file
                    
                    # Load image if requested
                    if self.load_images_in_memory:
                        image = cv2.imread(rgb_file)
                        if image is not None:
                            # Convert to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            self.images[camera_id] = image
                
                # Find semantic label if available
                if has_semantics:
                    sem_files = glob.glob(os.path.join(semantics_dir, f"{camera_name}_*.png"))
                    
                    if sem_files:
                        sem_file = sorted(sem_files)[0]  # Take first frame if multiple
                        label = cv2.imread(sem_file, cv2.IMREAD_UNCHANGED)
                        if label is not None:
                            self.semantics[camera_id] = label
                
            except Exception as e:
                logger.warning(f"Failed to process camera {camera_name}: {e}")
                continue
        
        # Update dataset info
        self.info.num_images = len(self.cameras)
        self.info.has_reconstruction = True
        self.info.has_gt_semantics = has_semantics
        
        logger.info(f"Initialized Hypersim dataset with {self.info.num_images} cameras")
        
        self.initialized = True
        return True