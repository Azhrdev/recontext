#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset processing script for RECONTEXT.

This script handles the preprocessing of various 3D reconstruction datasets,
converting them into a standard format that can be used by the RECONTEXT pipeline.
It supports multiple dataset formats including ScanNet, Matterport3D, and custom datasets.

Author: James Wei
Date: 2024-02-05
Last modified: 2024-03-15
"""

import os
import sys
import argparse
import logging
import shutil
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recontext.utils.io_utils import ensure_dir, load_images, save_images
from recontext.core.camera import Camera, CameraIntrinsics, CameraExtrinsics
from recontext.utils.colmap_utils import run_colmap, parse_colmap_output
from recontext.semantics.label_manager import LabelManager
from recontext.config.paths import get_data_dir, get_output_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('process_dataset.log', mode='w')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RECONTEXT Dataset Processing Tool")
    
    # Input dataset options
    parser.add_argument("--dataset_path", required=True, help="Path to input dataset")
    parser.add_argument("--dataset_type", choices=["scannet", "matterport3d", "custom", "colmap"],
                       default="custom", help="Type of dataset")
    parser.add_argument("--output_dir", help="Directory to save processed dataset")
    
    # Processing options
    parser.add_argument("--image_size", type=int, nargs=2, default=[640, 480],
                       help="Target image size (width height)")
    parser.add_argument("--subsample_images", type=int, default=0,
                       help="Subsample images (every Nth frame, 0 for no subsampling)")
    parser.add_argument("--extract_frames", action="store_true",
                       help="Extract frames from video if dataset contains videos")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames per second for video extraction")
    
    # Preprocessing options
    parser.add_argument("--run_colmap", action="store_true",
                       help="Run COLMAP for SfM if dataset does not have camera parameters")
    parser.add_argument("--skip_semantic", action="store_true",
                       help="Skip semantic label processing")
    parser.add_argument("--copy_pointcloud", action="store_true",
                       help="Copy point cloud from original dataset if available")
    
    # Label mapping
    parser.add_argument("--label_map", help="Path to label mapping JSON file")
    
    return parser.parse_args()

def process_scannet(dataset_path: str, output_dir: str, args) -> Dict:
    """Process ScanNet dataset.
    
    Args:
        dataset_path: Path to ScanNet dataset
        output_dir: Path to output directory
        args: Command line arguments
        
    Returns:
        Dictionary with dataset information
    """
    logger.info(f"Processing ScanNet dataset from {dataset_path}")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    poses_dir = os.path.join(output_dir, "poses")
    ensure_dir(images_dir)
    ensure_dir(poses_dir)
    
    # Expected directory structure
    color_dir = os.path.join(dataset_path, "color")
    depth_dir = os.path.join(dataset_path, "depth")
    pose_dir = os.path.join(dataset_path, "pose")
    
    # Check if directories exist
    if not os.path.exists(color_dir):
        logger.error(f"Color directory not found: {color_dir}")
        return {}
    
    # Get color image files
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg") or f.endswith(".png")])
    
    if args.subsample_images > 0:
        color_files = color_files[::args.subsample_images]
    
    logger.info(f"Found {len(color_files)} color images")
    
    # Process each image
    cameras = {}
    images_info = []
    
    for i, color_file in enumerate(tqdm(color_files, desc="Processing images")):
        image_id = i
        
        # Load color image
        color_path = os.path.join(color_dir, color_file)
        image = cv2.imread(color_path)
        
        if image is None:
            logger.warning(f"Failed to load image: {color_path}")
            continue
        
        # Resize if needed
        target_width, target_height = args.image_size
        if image.shape[1] != target_width or image.shape[0] != target_height:
            image = cv2.resize(image, (target_width, target_height))
        
        # Save color image
        output_image_path = os.path.join(images_dir, f"{image_id:06d}.jpg")
        cv2.imwrite(output_image_path, image)
        
        # Load corresponding depth image if available
        depth_file = color_file.replace(".jpg", ".png").replace(".png", ".png")
        depth_path = os.path.join(depth_dir, depth_file)
        
        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            # Resize if needed
            if depth.shape[1] != target_width or depth.shape[0] != target_height:
                depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            # Save depth image
            output_depth_path = os.path.join(images_dir, f"{image_id:06d}_depth.png")
            cv2.imwrite(output_depth_path, depth)
        
        # Load camera pose if available
        pose_file = color_file.replace(".jpg", ".txt").replace(".png", ".txt")
        pose_path = os.path.join(pose_dir, pose_file)
        
        if os.path.exists(pose_path):
            # Load camera extrinsics (4x4 matrix)
            extrinsic = np.loadtxt(pose_path)
            
            # Create camera intrinsics (ScanNet default)
            intrinsic = CameraIntrinsics(
                width=target_width,
                height=target_height,
                fx=577.870605 * (target_width / 640),
                fy=577.870605 * (target_height / 480),
                cx=319.5 * (target_width / 640),
                cy=239.5 * (target_height / 480)
            )
            
            # Create Camera object
            camera = Camera(
                intrinsics=intrinsic,
                extrinsics=CameraExtrinsics(
                    R=extrinsic[:3, :3],
                    t=extrinsic[:3, 3].reshape(3, 1)
                )
            )
            
            cameras[image_id] = camera
            
            # Save camera pose
            output_pose_path = os.path.join(poses_dir, f"{image_id:06d}.json")
            camera.save(output_pose_path)
        
        # Add to images info
        images_info.append({
            "id": image_id,
            "name": f"{image_id:06d}.jpg",
            "path": output_image_path
        })
    
    # Save dataset information
    dataset_info = {
        "name": os.path.basename(dataset_path),
        "type": "scannet",
        "images": images_info,
        "has_poses": len(cameras) > 0,
        "has_depth": os.path.exists(depth_dir),
        "image_size": args.image_size
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Copy point cloud if requested
    if args.copy_pointcloud:
        ply_path = os.path.join(dataset_path, "scene.ply")
        if os.path.exists(ply_path):
            shutil.copy(ply_path, os.path.join(output_dir, "pointcloud.ply"))
            logger.info(f"Copied point cloud from {ply_path}")
    
    # Process semantic labels if available and not skipped
    if not args.skip_semantic:
        label_dir = os.path.join(dataset_path, "label")
        if os.path.exists(label_dir):
            semantic_dir = os.path.join(output_dir, "semantic")
            ensure_dir(semantic_dir)
            
            # Load label mapping
            label_manager = LabelManager()
            if args.label_map:
                label_mapping = json.load(open(args.label_map, "r"))
            else:
                # Default ScanNet mapping
                label_mapping = label_manager.get_scannet_mapping()
            
            # Process each label file
            for i, color_file in enumerate(tqdm(color_files, desc="Processing semantic labels")):
                image_id = i
                
                # Load label image
                label_file = color_file.replace(".jpg", ".png").replace(".png", ".png")
                label_path = os.path.join(label_dir, label_file)
                
                if os.path.exists(label_path):
                    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
                    
                    # Resize if needed
                    if label.shape[1] != target_width or label.shape[0] != target_height:
                        label = cv2.resize(label, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                    
                    # Save label image
                    output_label_path = os.path.join(semantic_dir, f"{image_id:06d}_label.png")
                    cv2.imwrite(output_label_path, label)
            
            # Save label mapping
            with open(os.path.join(semantic_dir, "label_mapping.json"), "w") as f:
                json.dump(label_mapping, f, indent=2)
    
    logger.info(f"Dataset processing complete: {len(images_info)} images processed")
    return dataset_info

def process_matterport3d(dataset_path: str, output_dir: str, args) -> Dict:
    """Process Matterport3D dataset.
    
    Args:
        dataset_path: Path to Matterport3D dataset
        output_dir: Path to output directory
        args: Command line arguments
        
    Returns:
        Dictionary with dataset information
    """
    logger.info(f"Processing Matterport3D dataset from {dataset_path}")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    poses_dir = os.path.join(output_dir, "poses")
    ensure_dir(images_dir)
    ensure_dir(poses_dir)
    
    # Expected directory structure for Matterport3D
    # Get all house (scene) IDs
    house_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not house_dirs:
        logger.error(f"No house directories found in {dataset_path}")
        return {}
    
    # Get the first house (or specified one)
    house_id = house_dirs[0]
    logger.info(f"Processing house: {house_id}")
    
    # Directory paths
    house_dir = os.path.join(dataset_path, house_id)
    undistorted_dir = os.path.join(house_dir, "undistorted_color_images")
    
    if not os.path.exists(undistorted_dir):
        logger.error(f"Undistorted images directory not found: {undistorted_dir}")
        return {}
    
    # Get all viewpoints
    viewpoints = []
    for root, dirs, files in os.walk(undistorted_dir):
        for file in files:
            if file.endswith(".jpg"):
                viewpoints.append(os.path.join(root, file))
    
    if args.subsample_images > 0:
        viewpoints = viewpoints[::args.subsample_images]
    
    logger.info(f"Found {len(viewpoints)} viewpoints")
    
    # Process each viewpoint
    cameras = {}
    images_info = []
    
    for i, viewpoint in enumerate(tqdm(viewpoints, desc="Processing viewpoints")):
        image_id = i
        
        # Load image
        image = cv2.imread(viewpoint)
        
        if image is None:
            logger.warning(f"Failed to load image: {viewpoint}")
            continue
        
        # Resize if needed
        target_width, target_height = args.image_size
        if image.shape[1] != target_width or image.shape[0] != target_height:
            image = cv2.resize(image, (target_width, target_height))
        
        # Save image
        output_image_path = os.path.join(images_dir, f"{image_id:06d}.jpg")
        cv2.imwrite(output_image_path, image)
        
        # Extract relative path for finding pose
        rel_path = os.path.relpath(viewpoint, undistorted_dir)
        pose_file = rel_path.replace(".jpg", ".pose.txt")
        pose_path = os.path.join(house_dir, "undistorted_camera_poses", pose_file)
        
        # Load camera pose if available
        if os.path.exists(pose_path):
            # Matterport poses are 4x4 matrices
            extrinsic = np.loadtxt(pose_path)
            
            # Matterport intrinsics (default values, should be adjusted for specific dataset)
            intrinsic = CameraIntrinsics(
                width=target_width,
                height=target_height,
                fx=1080.0 * (target_width / 1920),
                fy=1080.0 * (target_height / 1440),
                cx=target_width / 2,
                cy=target_height / 2
            )
            
            # Create Camera object
            camera = Camera(
                intrinsics=intrinsic,
                extrinsics=CameraExtrinsics(
                    R=extrinsic[:3, :3],
                    t=extrinsic[:3, 3].reshape(3, 1)
                )
            )
            
            cameras[image_id] = camera
            
            # Save camera pose
            output_pose_path = os.path.join(poses_dir, f"{image_id:06d}.json")
            camera.save(output_pose_path)
        
        # Add to images info
        images_info.append({
            "id": image_id,
            "name": f"{image_id:06d}.jpg",
            "path": output_image_path,
            "original_path": viewpoint
        })
    
    # Save dataset information
    dataset_info = {
        "name": house_id,
        "type": "matterport3d",
        "images": images_info,
        "has_poses": len(cameras) > 0,
        "image_size": args.image_size
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Copy point cloud if requested
    if args.copy_pointcloud:
        ply_path = os.path.join(house_dir, "house_mesh.ply")
        if os.path.exists(ply_path):
            shutil.copy(ply_path, os.path.join(output_dir, "pointcloud.ply"))
            logger.info(f"Copied point cloud from {ply_path}")
    
    # Process semantic labels if available and not skipped
    if not args.skip_semantic:
        semantic_dir = os.path.join(output_dir, "semantic")
        ensure_dir(semantic_dir)
        
        # In Matterport3D, semantic information is in the region segmentations
        # This would require more specialized processing for Matterport3D
        
        # Load label mapping
        label_manager = LabelManager()
        if args.label_map:
            label_mapping = json.load(open(args.label_map, "r"))
        else:
            # Default Matterport3D mapping
            label_mapping = label_manager.get_matterport3d_mapping()
        
        # Save label mapping
        with open(os.path.join(semantic_dir, "label_mapping.json"), "w") as f:
            json.dump(label_mapping, f, indent=2)
    
    logger.info(f"Dataset processing complete: {len(images_info)} images processed")
    return dataset_info

def process_custom(dataset_path: str, output_dir: str, args) -> Dict:
    """Process custom dataset (directory of images and optional camera information).
    
    Args:
        dataset_path: Path to custom dataset
        output_dir: Path to output directory
        args: Command line arguments
        
    Returns:
        Dictionary with dataset information
    """
    logger.info(f"Processing custom dataset from {dataset_path}")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    poses_dir = os.path.join(output_dir, "poses")
    ensure_dir(images_dir)
    ensure_dir(poses_dir)
    
    # Check for images directory or use dataset_path directly
    images_path = os.path.join(dataset_path, "images")
    if not os.path.exists(images_path):
        images_path = dataset_path
    
    # Check for videos that need to be processed
    videos_path = os.path.join(dataset_path, "videos")
    if args.extract_frames and os.path.exists(videos_path):
        # Extract frames from videos
        extract_frames_from_videos(videos_path, images_dir, fps=args.fps)
        # Use extracted frames
        images_path = images_dir
    
    # Get image files
    image_files = sorted([f for f in os.listdir(images_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if args.subsample_images > 0:
        image_files = image_files[::args.subsample_images]
    
    logger.info(f"Found {len(image_files)} images")
    
    # Process each image
    images_info = []
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        image_id = i
        
        # Load image
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        # Resize if needed
        target_width, target_height = args.image_size
        if image.shape[1] != target_width or image.shape[0] != target_height:
            image = cv2.resize(image, (target_width, target_height))
        
        # Save image
        output_image_path = os.path.join(images_dir, f"{image_id:06d}.jpg")
        cv2.imwrite(output_image_path, image)
        
        # Add to images info
        images_info.append({
            "id": image_id,
            "name": f"{image_id:06d}.jpg",
            "path": output_image_path,
            "original_name": image_file
        })
    
    # Check for camera parameters
    cameras = {}
    has_poses = False
    
    # Look for camera parameters in different formats
    camera_files = []
    
    # Check for individual camera files
    camera_dir = os.path.join(dataset_path, "cameras")
    if os.path.exists(camera_dir):
        camera_files = [os.path.join(camera_dir, f) for f in os.listdir(camera_dir) 
                        if f.endswith(".json") or f.endswith(".txt")]
    
    # Check for COLMAP format
    colmap_dir = os.path.join(dataset_path, "colmap")
    if os.path.exists(colmap_dir):
        try:
            # Parse COLMAP output
            cameras, _ = parse_colmap_output(colmap_dir)
            has_poses = len(cameras) > 0
            
            # Save camera parameters
            for image_id, camera in cameras.items():
                output_pose_path = os.path.join(poses_dir, f"{image_id:06d}.json")
                camera.save(output_pose_path)
                
            logger.info(f"Loaded {len(cameras)} camera poses from COLMAP")
        except Exception as e:
            logger.error(f"Failed to parse COLMAP data: {e}")
    
    # Run COLMAP if requested and no camera parameters found
    if args.run_colmap and not has_poses:
        logger.info("Running COLMAP for SfM")
        
        colmap_output_dir = os.path.join(output_dir, "colmap")
        ensure_dir(colmap_output_dir)
        
        # Get image paths
        image_paths = [os.path.join(images_dir, info["name"]) for info in images_info]
        
        # Run COLMAP
        try:
            run_colmap(image_paths, colmap_output_dir, quality="medium")
            
            # Parse COLMAP output
            cameras, _ = parse_colmap_output(colmap_output_dir)
            has_poses = len(cameras) > 0
            
            # Save camera parameters
            for image_id, camera in cameras.items():
                output_pose_path = os.path.join(poses_dir, f"{image_id:06d}.json")
                camera.save(output_pose_path)
                
            logger.info(f"Generated {len(cameras)} camera poses with COLMAP")
        except Exception as e:
            logger.error(f"COLMAP processing failed: {e}")
    
    # Save dataset information
    dataset_info = {
        "name": os.path.basename(dataset_path),
        "type": "custom",
        "images": images_info,
        "has_poses": has_poses,
        "image_size": args.image_size
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Copy point cloud if requested
    if args.copy_pointcloud:
        ply_paths = [
            os.path.join(dataset_path, "pointcloud.ply"),
            os.path.join(dataset_path, "point_cloud.ply"),
            os.path.join(dataset_path, "mesh.ply")
        ]
        
        for ply_path in ply_paths:
            if os.path.exists(ply_path):
                shutil.copy(ply_path, os.path.join(output_dir, "pointcloud.ply"))
                logger.info(f"Copied point cloud from {ply_path}")
                break
    
    logger.info(f"Dataset processing complete: {len(images_info)} images processed")
    return dataset_info

def process_colmap(dataset_path: str, output_dir: str, args) -> Dict:
    """Process dataset in COLMAP format.
    
    Args:
        dataset_path: Path to COLMAP dataset
        output_dir: Path to output directory
        args: Command line arguments
        
    Returns:
        Dictionary with dataset information
    """
    logger.info(f"Processing COLMAP dataset from {dataset_path}")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    poses_dir = os.path.join(output_dir, "poses")
    ensure_dir(images_dir)
    ensure_dir(poses_dir)
    
    # Check for COLMAP directories
    colmap_images_dir = os.path.join(dataset_path, "images")
    colmap_sparse_dir = os.path.join(dataset_path, "sparse")
    
    if not os.path.exists(colmap_images_dir):
        logger.error(f"COLMAP images directory not found: {colmap_images_dir}")
        return {}
    
    if not os.path.exists(colmap_sparse_dir):
        logger.error(f"COLMAP sparse directory not found: {colmap_sparse_dir}")
        return {}
    
    # Get image files
    image_files = sorted([f for f in os.listdir(colmap_images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if args.subsample_images > 0:
        image_files = image_files[::args.subsample_images]
    
    logger.info(f"Found {len(image_files)} images")
    
    # Process images
    images_info = []
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        image_id = i
        
        # Load image
        image_path = os.path.join(colmap_images_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            continue
        
        # Resize if needed
        target_width, target_height = args.image_size
        if image.shape[1] != target_width or image.shape[0] != target_height:
            image = cv2.resize(image, (target_width, target_height))
        
        # Save image
        output_image_path = os.path.join(images_dir, f"{image_id:06d}.jpg")
        cv2.imwrite(output_image_path, image)
        
        # Add to images info
        images_info.append({
            "id": image_id,
            "name": f"{image_id:06d}.jpg",
            "path": output_image_path,
            "original_name": image_file
        })
    
    # Parse COLMAP output to get camera parameters
    try:
        # Find the most recent COLMAP model
        colmap_model_dirs = [d for d in os.listdir(colmap_sparse_dir) 
                            if os.path.isdir(os.path.join(colmap_sparse_dir, d))]
        
        if not colmap_model_dirs:
            raise FileNotFoundError("No COLMAP model directory found")
        
        # Use the first model directory (or 0 directory if available)
        model_dir = "0" if "0" in colmap_model_dirs else colmap_model_dirs[0]
        colmap_model_path = os.path.join(colmap_sparse_dir, model_dir)
        
        # Parse COLMAP output
        cameras, _ = parse_colmap_output(colmap_model_path)
        has_poses = len(cameras) > 0
        
        # Remap image IDs to sequential order
        remapped_cameras = {}
        for i, (orig_id, orig_camera) in enumerate(cameras.items()):
            remapped_cameras[i] = orig_camera
        
        # Save camera parameters
        for image_id, camera in remapped_cameras.items():
            output_pose_path = os.path.join(poses_dir, f"{image_id:06d}.json")
            camera.save(output_pose_path)
            
        logger.info(f"Loaded {len(remapped_cameras)} camera poses from COLMAP")
        
    except Exception as e:
        logger.error(f"Failed to parse COLMAP data: {e}")
        has_poses = False
    
    # Save dataset information
    dataset_info = {
        "name": os.path.basename(dataset_path),
        "type": "colmap",
        "images": images_info,
        "has_poses": has_poses,
        "image_size": args.image_size
    }
    
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    # Copy point cloud if requested
    if args.copy_pointcloud:
        # Check for dense point cloud
        dense_dir = os.path.join(dataset_path, "dense")
        if os.path.exists(dense_dir):
            for subdir in os.listdir(dense_dir):
                fused_path = os.path.join(dense_dir, subdir, "fused.ply")
                if os.path.exists(fused_path):
                    shutil.copy(fused_path, os.path.join(output_dir, "pointcloud.ply"))
                    logger.info(f"Copied point cloud from {fused_path}")
                    break
    
    logger.info(f"Dataset processing complete: {len(images_info)} images processed")
    return dataset_info

def extract_frames_from_videos(videos_path: str, output_dir: str, fps: int = 2) -> List[str]:
    """Extract frames from videos.
    
    Args:
        videos_path: Path to directory with videos
        output_dir: Path to output directory
        fps: Frames per second to extract
        
    Returns:
        List of extracted frame paths
    """
    logger.info(f"Extracting frames from videos in {videos_path}")
    ensure_dir(output_dir)
    
    # Get video files
    video_files = [f for f in os.listdir(videos_path) 
                 if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        logger.warning(f"No video files found in {videos_path}")
        return []
    
    logger.info(f"Found {len(video_files)} video files")
    
    # Extract frames from each video
    frame_paths = []
    frame_count = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(videos_path, video_file)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Failed to open video: {video_path}")
            continue
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            logger.warning(f"Invalid FPS for video: {video_path}")
            continue
        
        # Calculate frame step to achieve desired FPS
        frame_step = int(video_fps / fps)
        if frame_step < 1:
            frame_step = 1
        
        # Extract frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame if it's time
            if frame_idx % frame_step == 0:
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                
                frame_paths.append(frame_path)
                frame_count += 1
            
            frame_idx += 1
        
        # Release video
        cap.release()
    
    logger.info(f"Extracted {len(frame_paths)} frames from {len(video_files)} videos")
    return frame_paths

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Get output directory
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(get_output_dir(), "datasets", os.path.basename(args.dataset_path))
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Process dataset based on type
    if args.dataset_type == "scannet":
        dataset_info = process_scannet(args.dataset_path, output_dir, args)
    elif args.dataset_type == "matterport3d":
        dataset_info = process_matterport3d(args.dataset_path, output_dir, args)
    elif args.dataset_type == "colmap":
        dataset_info = process_colmap(args.dataset_path, output_dir, args)
    else:  # custom
        dataset_info = process_custom(args.dataset_path, output_dir, args)
    
    # Check if we have processed anything
    if not dataset_info:
        logger.error("Dataset processing failed")
        return 1
    
    logger.info(f"Dataset processing complete. Output in: {output_dir}")
    logger.info(f"Dataset summary:")
    logger.info(f"  Name: {dataset_info.get('name', 'Unknown')}")
    logger.info(f"  Type: {dataset_info.get('type', 'Unknown')}")
    logger.info(f"  Images: {len(dataset_info.get('images', []))}")
    logger.info(f"  Has camera poses: {dataset_info.get('has_poses', False)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())