#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature extraction module supporting different feature detectors and descriptors.
Implements both traditional (SIFT, ORB) and learning-based (SuperPoint) methods.

Author: Alex Johnson
Date: 2024-01-18
Last modified: 2024-03-12
"""

import os
import torch
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from tqdm import tqdm

from recontext.utils.io_utils import download_model

logger = logging.getLogger(__name__)

@dataclass
class FeatureData:
    """Container for feature data."""
    keypoints: np.ndarray  # Nx2 array of (x, y) coordinates
    descriptors: np.ndarray  # NxD array of descriptors
    scores: np.ndarray  # N array of detection scores/responses
    image_size: Tuple[int, int]  # (width, height) of the source image
    feature_type: str  # Type of features (e.g., 'sift', 'superpoint')
    
    @property
    def num_features(self) -> int:
        """Get number of features."""
        return len(self.keypoints)
    
    def to_cv_keypoints(self) -> List[cv2.KeyPoint]:
        """Convert to OpenCV KeyPoint objects."""
        keypoints = []
        for i, (x, y) in enumerate(self.keypoints):
            # Create KeyPoint with position, size, response
            kp = cv2.KeyPoint(
                x=float(x),
                y=float(y),
                size=1.0,  # Default size
                angle=-1,  # No orientation
                response=float(self.scores[i]),
                octave=0,
                class_id=-1
            )
            keypoints.append(kp)
        
        return keypoints


class FeatureExtractor:
    """Base class for feature extractors."""
    
    def __init__(self, name: str):
        """Initialize feature extractor.
        
        Args:
            name: Name of the feature extractor
        """
        self.name = name
    
    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract features from image.
        
        Args:
            image: Input image
            
        Returns:
            Feature data
        """
        raise NotImplementedError("Subclasses must implement extract")
    
    def extract_batch(self, images: List[np.ndarray]) -> List[FeatureData]:
        """Extract features from multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of feature data
        """
        features = []
        for image in tqdm(images, desc=f"Extracting {self.name} features"):
            features.append(self.extract(image))
        
        return features


class SIFTExtractor(FeatureExtractor):
    """SIFT feature extractor."""
    
    def __init__(self, 
                 max_features: int = 2000,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10):
        """Initialize SIFT extractor.
        
        Args:
            max_features: Maximum number of features
            contrast_threshold: Contrast threshold
            edge_threshold: Edge threshold
        """
        super().__init__("sift")
        
        self.max_features = max_features
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        
        # Create SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=self.max_features,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold
        )
    
    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract SIFT features.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            SIFT feature data
        """
        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        
        # Check if features were found
        if keypoints is None or len(keypoints) == 0:
            # Return empty feature data
            return FeatureData(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 128), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                image_size=(image.shape[1], image.shape[0]),
                feature_type="sift"
            )
        
        # Convert keypoints to numpy array
        kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        
        # Get response scores
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        
        return FeatureData(
            keypoints=kp_array,
            descriptors=descriptors,
            scores=scores,
            image_size=(image.shape[1], image.shape[0]),
            feature_type="sift"
        )


class ORBExtractor(FeatureExtractor):
    """ORB feature extractor."""
    
    def __init__(self, max_features: int = 2000, scale_factor: float = 1.2):
        """Initialize ORB extractor.
        
        Args:
            max_features: Maximum number of features
            scale_factor: Scale factor for multi-scale detection
        """
        super().__init__("orb")
        
        self.max_features = max_features
        self.scale_factor = scale_factor
        
        # Create ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=self.max_features,
            scaleFactor=self.scale_factor
        )
    
    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract ORB features.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            ORB feature data
        """
        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(image, None)
        
        # Check if features were found
        if keypoints is None or len(keypoints) == 0:
            # Return empty feature data
            return FeatureData(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, 32), dtype=np.uint8),
                scores=np.zeros(0, dtype=np.float32),
                image_size=(image.shape[1], image.shape[0]),
                feature_type="orb"
            )
        
        # Convert keypoints to numpy array
        kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
        
        # Get response scores
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        
        # Ensure descriptors array exists
        if descriptors is None:
            descriptors = np.zeros((len(keypoints), 32), dtype=np.uint8)
        
        return FeatureData(
            keypoints=kp_array,
            descriptors=descriptors,
            scores=scores,
            image_size=(image.shape[1], image.shape[0]),
            feature_type="orb"
        )


class SuperPointExtractor(FeatureExtractor):
    """SuperPoint feature extractor based on learned features."""
    
    MODEL_URLS = {
        'superpoint': 'https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/superpoint_v1.pth',
    }
    
    def __init__(self, 
                 max_features: int = 2000,
                 detection_threshold: float = 0.005,
                 nms_radius: int = 4,
                 device: Optional[str] = None):
        """Initialize SuperPoint extractor.
        
        Args:
            max_features: Maximum number of features
            detection_threshold: Keypoint detection threshold
            nms_radius: Non-maximum suppression radius
            device: Device to use for inference ('cuda' or 'cpu')
        """
        super().__init__("superpoint")
        
        self.max_features = max_features
        self.detection_threshold = detection_threshold
        self.nms_radius = nms_radius
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load SuperPoint model."""
        try:
            # Try to import SuperPoint
            from models.superpoint import SuperPoint
            
            # Create model
            config = {
                'descriptor_dim': 256,
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.detection_threshold,
                'max_keypoints': self.max_features,
                'remove_borders': 4
            }
            
            model = SuperPoint(config)
            model.eval()
            
            # Download weights if needed
            weights_path = self._get_model_path()
            
            # Load weights
            weights = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(weights)
            
            # Move to device
            model = model.to(self.device)
            
            logger.info("SuperPoint model loaded successfully")
            
            return model
            
        except ImportError:
            # Fallback to minimal SuperPoint implementation
            logger.warning("SuperPoint package not found, using minimal implementation")
            
            return MinimalSuperPoint(
                max_keypoints=self.max_features,
                keypoint_threshold=self.detection_threshold,
                nms_radius=self.nms_radius,
                device=self.device
            )
    
    def _get_model_path(self) -> str:
        """Get path to model weights, downloading if necessary."""
        # Create cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".recontext", "models")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine model filename
        model_url = self.MODEL_URLS['superpoint']
        filename = os.path.basename(model_url)
        model_path = os.path.join(cache_dir, filename)
        
        # Download if needed
        if not os.path.exists(model_path):
            logger.info(f"Downloading SuperPoint model...")
            download_model(model_url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return model_path
    
    def extract(self, image: np.ndarray) -> FeatureData:
        """Extract SuperPoint features.
        
        Args:
            image: Input image
            
        Returns:
            SuperPoint feature data
        """
        # Ensure image is grayscale
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Normalize image
        img_normalized = gray.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            output = self.model({'image': img_tensor})
        
        # Get keypoints and descriptors
        keypoints = output['keypoints'][0].cpu().numpy()
        scores = output['scores'][0].cpu().numpy()
        descriptors = output['descriptors'][0].cpu().numpy().transpose()
        
        # Sort by score (descending) and limit number of features
        if len(scores) > self.max_features:
            indices = np.argsort(scores)[::-1][:self.max_features]
            keypoints = keypoints[indices]
            scores = scores[indices]
            descriptors = descriptors[indices]
        
        return FeatureData(
            keypoints=keypoints,
            descriptors=descriptors,
            scores=scores,
            image_size=(image.shape[1], image.shape[0]),
            feature_type="superpoint"
        )


class MinimalSuperPoint(torch.nn.Module):
    """Minimal SuperPoint implementation for feature extraction."""
    
    def __init__(self, 
                 max_keypoints: int = 1000,
                 keypoint_threshold: float = 0.005,
                 nms_radius: int = 4,
                 device: torch.device = None):
        """Initialize minimal SuperPoint model.
        
        Args:
            max_keypoints: Maximum number of keypoints
            keypoint_threshold: Keypoint detection threshold
            nms_radius: Non-maximum suppression radius
            device: Device to use
        """
        super().__init__()
        
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        self.device = device
        
        # Create layers
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        
        # Detector head
        self.detector = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        )
        
        # Descriptor head
        self.descriptor = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        
        # Move to device
        self.to(device)
    
    def forward(self, data: Dict) -> Dict:
        """Forward pass.
        
        Args:
            data: Dictionary with 'image' key containing input tensor
            
        Returns:
            Dictionary with 'keypoints', 'scores', and 'descriptors' keys
        """
        # Get input image
        image = data['image']
        
        # Shared encoder
        features = self.encoder(image)
        
        # Detector head
        heatmap = self.detector(features)
        
        # Get keypoints from heatmap
        keypoints, scores = self._extract_keypoints(heatmap)
        
        # Descriptor head
        descriptors = self.descriptor(features)
        
        # Sample descriptors at keypoint locations
        descriptors = self._sample_descriptors(descriptors, keypoints)
        
        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors
        }
    
    def _extract_keypoints(self, heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract keypoints from heatmap.
        
        Args:
            heatmap: Heatmap tensor
            
        Returns:
            Tuple of (keypoints, scores)
        """
        batch_size = heatmap.shape[0]
        
        # Reshape and softmax
        prob = torch.nn.functional.softmax(heatmap.view(batch_size, 65, -1), dim=1)
        prob = prob.view(batch_size, 65, heatmap.shape[2], heatmap.shape[3])
        
        # Get keypoint map (remove dustbin channel)
        keypoint_map = prob[:, :-1, :, :]
        
        # Apply non-maximum suppression
        keypoint_map = self._nms(keypoint_map)
        
        # Extract keypoints
        keypoints = []
        scores = []
        
        for i in range(batch_size):
            kmap = keypoint_map[i].view(-1)
            
            # Apply threshold
            mask = kmap > self.keypoint_threshold
            kmap = kmap[mask]
            
            # Get coordinates
            indices = torch.nonzero(mask).squeeze()
            y = indices // keypoint_map.shape[3]
            x = indices % keypoint_map.shape[3]
            
            # Convert to original image coordinates
            scale = 8.0  # Downsampling factor
            x = x.float() * scale + scale / 2
            y = y.float() * scale + scale / 2
            
            # Stack coordinates
            coords = torch.stack([x, y], dim=1)
            
            # Limit number of keypoints
            if len(kmap) > self.max_keypoints:
                _, indices = torch.topk(kmap, self.max_keypoints)
                coords = coords[indices]
                kmap = kmap[indices]
            
            keypoints.append(coords)
            scores.append(kmap)
        
        return keypoints, scores
    
    def _nms(self, heatmap: torch.Tensor) -> torch.Tensor:
        """Apply non-maximum suppression to heatmap.
        
        Args:
            heatmap: Heatmap tensor
            
        Returns:
            Filtered heatmap
        """
        # Get max in local neighborhood
        max_pool = torch.nn.functional.max_pool2d(
            heatmap, kernel_size=self.nms_radius * 2 + 1, stride=1, padding=self.nms_radius)
        
        # Keep only pixels that are local maxima
        return heatmap * (heatmap == max_pool)
    
    def _sample_descriptors(self, descriptors: torch.Tensor, keypoints: List[torch.Tensor]) -> List[torch.Tensor]:
        """Sample descriptors at keypoint locations.
        
        Args:
            descriptors: Descriptor tensor
            keypoints: List of keypoint coordinates
            
        Returns:
            List of descriptors
        """
        batch_size = descriptors.shape[0]
        result = []
        
        for i in range(batch_size):
            kpts = keypoints[i]
            
            if len(kpts) == 0:
                desc = torch.zeros((0, descriptors.shape[1]), device=self.device)
                result.append(desc)
                continue
            
            # Normalize coordinates to [-1, 1]
            h, w = descriptors.shape[2], descriptors.shape[3]
            kpts_norm = kpts.clone()
            kpts_norm[:, 0] = 2 * kpts[:, 0] / (w * 8 - 1) - 1
            kpts_norm[:, 1] = 2 * kpts[:, 1] / (h * 8 - 1) - 1
            
            # Reshape for grid_sample
            kpts_norm = kpts_norm.view(1, 1, -1, 2)
            
            # Sample descriptors
            desc = torch.nn.functional.grid_sample(descriptors[i:i+1], kpts_norm, align_corners=True)
            
            # Reshape and normalize
            desc = desc.view(descriptors.shape[1], -1).t()
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
            
            result.append(desc)
        
        return result


def create_feature_extractor(feature_type: str, **kwargs) -> FeatureExtractor:
    """Factory function for creating feature extractors.
    
    Args:
        feature_type: Type of feature extractor ('sift', 'orb', 'superpoint')
        **kwargs: Additional parameters for the extractor
        
    Returns:
        Feature extractor instance
    """
    if feature_type == 'sift':
        return SIFTExtractor(**kwargs)
    elif feature_type == 'orb':
        return ORBExtractor(**kwargs)
    elif feature_type == 'superpoint':
        return SuperPointExtractor(**kwargs)
    else:
        logger.warning(f"Unknown feature type: {feature_type}, using SIFT")
        return SIFTExtractor(**kwargs)


def extract_features(images: List[np.ndarray], 
                    feature_type: str = 'sift',
                    **kwargs) -> List[FeatureData]:
    """Extract features from images.
    
    Args:
        images: List of input images
        feature_type: Type of features to extract
        **kwargs: Additional parameters for the extractor
        
    Returns:
        List of feature data
    """
    # Create feature extractor
    extractor = create_feature_extractor(feature_type, **kwargs)
    
    # Extract features
    logger.info(f"Extracting {feature_type} features from {len(images)} images")
    features = extractor.extract_batch(images)
    
    # Log statistics
    total_features = sum(f.num_features for f in features)
    avg_features = total_features / len(images) if images else 0
    logger.info(f"Extracted {total_features} features ({avg_features:.1f} per image)")
    
    return features


def draw_features(image: np.ndarray, features: FeatureData) -> np.ndarray:
    """Draw features on image.
    
    Args:
        image: Input image
        features: Feature data
        
    Returns:
        Image with drawn features
    """
    # Convert to BGR if grayscale
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image.copy()
    
    # Convert to OpenCV keypoints
    keypoints = features.to_cv_keypoints()
    
    # Draw keypoints
    image_with_kp = cv2.drawKeypoints(
        image_bgr, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0)
    )
    
    return image_with_kp


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature extraction example")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--feature_type", choices=["sift", "orb", "superpoint"],
                       default="sift", help="Feature type")
    parser.add_argument("--max_features", type=int, default=2000,
                       help="Maximum number of features")
    parser.add_argument("--output", default="features.jpg",
                       help="Output visualization")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"Failed to load image: {args.image}")
        return
    
    # Extract features
    features = extract_features(
        [image], 
        feature_type=args.feature_type,
        max_features=args.max_features
    )[0]
    
    logger.info(f"Extracted {features.num_features} {args.feature_type} features")
    
    # Draw features
    visualization = draw_features(image, features)
    
    # Save visualization
    cv2.imwrite(args.output, visualization)
    logger.info(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()