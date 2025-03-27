#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I/O utility functions for loading and saving data, downloading models, etc.

Author: Michael Chen
Date: 2024-01-15
Last modified: 2024-03-02
"""

import os
import sys
import requests
import numpy as np
import cv2
import logging
import json
import pickle
import shutil
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any, BinaryIO
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, block_num: int, block_size: int, total_size: int):
        """Update progress bar.
        
        Args:
            block_num: Number of blocks transferred
            block_size: Size of each block in bytes
            total_size: Total size in bytes
        """
        if total_size <= 0:  # Unknown size
            self.total = None
        else:
            self.total = total_size
        
        self.update(block_size)


def download_file(url: str, output_path: str) -> str:
    """Download file with progress bar.
    
    Args:
        url: URL to download
        output_path: Path to save file
        
    Returns:
        Path to downloaded file
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    return output_path


def download_model(url: str, output_path: str) -> str:
    """Download pre-trained model.
    
    Args:
        url: URL to download
        output_path: Path to save model
        
    Returns:
        Path to downloaded model
    """
    logger.info(f"Downloading model from {url}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        download_file(url, output_path)
        logger.info(f"Model downloaded to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def ensure_dir(directory: str) -> str:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Absolute path to directory
    """
    if directory == "":
        return directory
        
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    return directory


def load_images(image_paths: List[str], 
               target_size: Optional[Tuple[int, int]] = None,
               grayscale: bool = False) -> List[np.ndarray]:
    """Load images from file paths.
    
    Args:
        image_paths: List of image file paths
        target_size: Optional target size for resizing (width, height)
        grayscale: Whether to convert images to grayscale
        
    Returns:
        List of loaded images
    """
    images = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        # Load image
        img = cv2.imread(path)
        
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            continue
        
        # Convert to grayscale if requested
        if grayscale:
            if len(img.shape) > 2 and img.shape[2] > 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif len(img.shape) == 2:  # Convert grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize if requested
        if target_size is not None:
            img = cv2.resize(img, target_size)
        
        images.append(img)
    
    return images


def save_images(images: List[np.ndarray], 
               output_dir: str, 
               prefix: str = "image",
               format: str = "jpg") -> List[str]:
    """Save images to files.
    
    Args:
        images: List of images
        output_dir: Output directory
        prefix: Filename prefix
        format: Image format
        
    Returns:
        List of saved file paths
    """
    # Create output directory
    ensure_dir(output_dir)
    
    # Save images
    file_paths = []
    for i, img in enumerate(tqdm(images, desc="Saving images")):
        # Generate filename
        filename = f"{prefix}_{i:04d}.{format}"
        path = os.path.join(output_dir, filename)
        
        # Save image
        cv2.imwrite(path, img)
        
        file_paths.append(path)
    
    return file_paths


def load_json(filepath: str) -> Dict:
    """Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON file {filepath}: {e}")
        raise


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        logger.error(f"Failed to save JSON file {filepath}: {e}")
        raise


def save_reconstruction(data: Dict, output_dir: str) -> Dict[str, str]:
    """Save reconstruction results to files.
    
    Args:
        data: Dictionary of reconstruction results
        output_dir: Output directory
        
    Returns:
        Dictionary of saved file paths
    """
    # Create output directory
    ensure_dir(output_dir)
    
    # File paths
    file_paths = {}
    
    # Save cameras
    if 'cameras' in data:
        cameras_path = os.path.join(output_dir, 'cameras.pkl')
        with open(cameras_path, 'wb') as f:
            pickle.dump(data['cameras'], f)
        file_paths['cameras'] = cameras_path
    
    # Save sparse point cloud
    if 'sparse_pointcloud' in data and data['sparse_pointcloud'] is not None:
        sparse_path = os.path.join(output_dir, 'sparse_pointcloud.ply')
        o3d_installed = False
        
        try:
            import open3d as o3d
            o3d_installed = True
            
            o3d.io.write_point_cloud(sparse_path, data['sparse_pointcloud'])
            file_paths['sparse_pointcloud'] = sparse_path
            
        except ImportError:
            logger.warning("open3d not installed, skipping point cloud saving")
        except Exception as e:
            logger.error(f"Failed to save sparse point cloud: {e}")
    
    # Save dense point cloud
    if 'dense_pointcloud' in data and data['dense_pointcloud'] is not None:
        dense_path = os.path.join(output_dir, 'dense_pointcloud.ply')
        
        if o3d_installed:
            try:
                o3d.io.write_point_cloud(dense_path, data['dense_pointcloud'])
                file_paths['dense_pointcloud'] = dense_path
            except Exception as e:
                logger.error(f"Failed to save dense point cloud: {e}")
    
    # Save mesh
    if 'mesh' in data and data['mesh'] is not None:
        mesh_path = os.path.join(output_dir, 'mesh.ply')
        
        if o3d_installed:
            try:
                o3d.io.write_triangle_mesh(mesh_path, data['mesh'])
                file_paths['mesh'] = mesh_path
            except Exception as e:
                logger.error(f"Failed to save mesh: {e}")
    
    # Save metadata
    metadata = {
        'timestamp': get_timestamp(),
        'file_paths': file_paths
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    save_json(metadata, metadata_path)
    
    return file_paths


def get_timestamp() -> str:
    """Get current timestamp string.
    
    Returns:
        Timestamp string in format "YYYY-MM-DD_HH-MM-SS"
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def copy_to_output_dir(filepath: str, output_dir: str, new_name: Optional[str] = None) -> str:
    """Copy file to output directory.
    
    Args:
        filepath: Path to source file
        output_dir: Output directory
        new_name: Optional new filename
        
    Returns:
        Path to copied file
    """
    # Create output directory
    ensure_dir(output_dir)
    
    # Determine output path
    if new_name is None:
        new_name = os.path.basename(filepath)
    
    output_path = os.path.join(output_dir, new_name)
    
    # Copy file
    shutil.copy2(filepath, output_path)
    
    return output_path


def file_list_handler(file_handler: callable) -> callable:
    """Decorator for handling functions that accept file paths or lists of file paths.
    
    Args:
        file_handler: Function that handles a single file
        
    Returns:
        Function that handles single file or list of files
    """
    def wrapper(file_paths_or_list, *args, **kwargs):
        """Handle single file or list of files.
        
        Args:
            file_paths_or_list: Single file path or list of file paths
            *args: Additional arguments for file handler
            **kwargs: Additional keyword arguments for file handler
            
        Returns:
            Result of file handler (single result or list of results)
        """
        if isinstance(file_paths_or_list, (list, tuple)):
            return [file_handler(path, *args, **kwargs) for path in file_paths_or_list]
        else:
            return file_handler(file_paths_or_list, *args, **kwargs)
    
    return wrapper


@file_list_handler
def check_file_exists(filepath: str) -> bool:
    """Check if file exists.
    
    Args:
        filepath: Path to file
        
    Returns:
        True if file exists, False otherwise
    """
    return os.path.isfile(filepath)


def get_file_extension(filepath: str) -> str:
    """Get file extension.
    
    Args:
        filepath: Path to file
        
    Returns:
        File extension (lowercase, without dot)
    """
    return os.path.splitext(filepath)[1].lower().lstrip('.')


def get_file_basename(filepath: str, with_extension: bool = True) -> str:
    """Get file basename.
    
    Args:
        filepath: Path to file
        with_extension: Whether to include extension
        
    Returns:
        File basename
    """
    basename = os.path.basename(filepath)
    if not with_extension:
        basename = os.path.splitext(basename)[0]
    
    return basename


def get_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
    """Get all files with specified extensions in directory.
    
    Args:
        directory: Directory path
        extensions: List of file extensions (without dots)
        
    Returns:
        List of file paths
    """
    # Normalize extensions
    extensions = [ext.lower().lstrip('.') for ext in extensions]
    
    # Get files
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if get_file_extension(file) in extensions:
                file_paths.append(os.path.join(root, file))
    
    # Sort files
    file_paths.sort()
    
    return file_paths


def get_image_files(directory: str) -> List[str]:
    """Get all image files in directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of image file paths
    """
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    return get_files_by_extension(directory, image_extensions)


def extract_frame_from_video(video_path: str, 
                            frame_number: int, 
                            output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Extract frame from video.
    
    Args:
        video_path: Path to video file
        frame_number: Frame number to extract
        output_path: Optional path to save extracted frame
        
    Returns:
        Extracted frame or None if failed
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Get frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if frame number is valid
        if frame_number < 0 or frame_number >= frame_count:
            logger.error(f"Invalid frame number: {frame_number} (total frames: {frame_count})")
            cap.release()
            return None
        
        # Set position to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = cap.read()
        
        # Release video
        cap.release()
        
        if not ret:
            logger.error(f"Failed to read frame {frame_number} from video")
            return None
        
        # Save frame if output path is provided
        if output_path is not None:
            cv2.imwrite(output_path, frame)
        
        return frame
        
    except Exception as e:
        logger.error(f"Failed to extract frame from video: {e}")
        return None


def extract_frames_from_video(video_path: str, 
                             output_dir: str,
                             frame_step: int = 1,
                             max_frames: Optional[int] = None,
                             output_format: str = 'jpg') -> List[str]:
    """Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory
        frame_step: Number of frames to skip between extractions
        max_frames: Maximum number of frames to extract
        output_format: Output image format
        
    Returns:
        List of extracted frame paths
    """
    try:
        # Create output directory
        ensure_dir(output_dir)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video: {get_file_basename(video_path)}")
        logger.info(f"Frames: {frame_count}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
        
        # Determine number of frames to extract
        if max_frames is not None:
            # Adjust frame_step to extract at most max_frames
            min_step = max(1, frame_count // max_frames)
            frame_step = max(frame_step, min_step)
        
        # Extract frames
        frame_paths = []
        frame_idx = 0
        
        # Create progress bar
        pbar = tqdm(total=frame_count, desc="Extracting frames")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame if it's time
            if frame_idx % frame_step == 0:
                # Generate output path
                output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.{output_format}")
                
                # Save frame
                cv2.imwrite(output_path, frame)
                
                frame_paths.append(output_path)
                
                # Check if we've reached the maximum number of frames
                if max_frames is not None and len(frame_paths) >= max_frames:
                    break
            
            # Update frame index and progress bar
            frame_idx += 1
            pbar.update(1)
        
        # Close progress bar and release video
        pbar.close()
        cap.release()
        
        logger.info(f"Extracted {len(frame_paths)} frames")
        
        return frame_paths
        
    except Exception as e:
        logger.error(f"Failed to extract frames from video: {e}")
        return []


def is_valid_image(image_path: str) -> bool:
    """Check if image is valid.
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        img = cv2.imread(image_path)
        return img is not None and img.size > 0
    except Exception:
        return False


def is_valid_video(video_path: str) -> bool:
    """Check if video is valid.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        valid = cap.isOpened()
        cap.release()
        return valid
    except Exception:
        return False


def get_image_info(image_path: str) -> Dict:
    """Get image information.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image information
    """
    try:
        img = cv2.imread(image_path)
        
        if img is None:
            return {'valid': False}
        
        height, width = img.shape[:2]
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        file_size = os.path.getsize(image_path)
        
        return {
            'valid': True,
            'path': image_path,
            'width': width,
            'height': height,
            'channels': channels,
            'file_size': file_size,
            'file_type': get_file_extension(image_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to get image info: {e}")
        return {'valid': False, 'path': image_path, 'error': str(e)}


def get_video_info(video_path: str) -> Dict:
    """Get video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            cap.release()
            return {'valid': False}
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        file_size = os.path.getsize(video_path)
        
        cap.release()
        
        return {
            'valid': True,
            'path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'file_size': file_size,
            'file_type': get_file_extension(video_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {'valid': False, 'path': video_path, 'error': str(e)}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        # Process command line arguments
        if sys.argv[1] == "--extract-frames" and len(sys.argv) >= 4:
            video_path = sys.argv[2]
            output_dir = sys.argv[3]
            frame_step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
            max_frames = int(sys.argv[5]) if len(sys.argv) > 5 else None
            
            frame_paths = extract_frames_from_video(
                video_path, output_dir, frame_step, max_frames)
                
            logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
            
        elif sys.argv[1] == "--get-info":
            for path in sys.argv[2:]:
                if is_valid_image(path):
                    info = get_image_info(path)
                    logger.info(f"Image: {path}")
                    logger.info(f"  Size: {info['width']}x{info['height']}")
                    logger.info(f"  Channels: {info['channels']}")
                    logger.info(f"  File size: {info['file_size'] / 1024:.1f} KB")
                    
                elif is_valid_video(path):
                    info = get_video_info(path)
                    logger.info(f"Video: {path}")
                    logger.info(f"  Size: {info['width']}x{info['height']}")
                    logger.info(f"  Duration: {info['duration']:.2f} seconds")
                    logger.info(f"  Frames: {info['frame_count']} @ {info['fps']:.2f} FPS")
                    logger.info(f"  File size: {info['file_size'] / (1024*1024):.1f} MB")
                    
                else:
                    logger.error(f"Invalid image or video: {path}")
            
        else:
            logger.info("Usage:")
            logger.info("  --extract-frames <video_path> <output_dir> [frame_step] [max_frames]")
            logger.info("  --get-info <path1> [path2] [...]")
    else:
        logger.info("No arguments provided. Examples:")
        logger.info("  python io_utils.py --extract-frames video.mp4 frames/ 10 100")
        logger.info("  python io_utils.py --get-info image.jpg video.mp4")