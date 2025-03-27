#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Feature matching module supporting different matching algorithms.
Implements both traditional (mutual nearest neighbors, FLANN) and 
learning-based (SuperGlue) matching methods.

Author: Alex Johnson
Date: 2024-01-20
Last modified: 2024-03-15
"""

import os
import numpy as np
import cv2
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from tqdm import tqdm

from recontext.core.feature_extraction import FeatureData
from recontext.utils.io_utils import download_model

logger = logging.getLogger(__name__)

@dataclass
class MatchData:
    """Container for feature matching data."""
    image_pair: Tuple[int, int]  # Pair of image indices (i, j) where i < j
    matches: np.ndarray  # Mx2 array of feature indices (idx_i, idx_j)
    confidence: np.ndarray  # M array of match confidence scores
    match_type: str  # Type of matching algorithm used
    
    @property
    def num_matches(self) -> int:
        """Get number of matches."""
        return len(self.matches)
    
    def filter_by_confidence(self, threshold: float) -> 'MatchData':
        """Filter matches by confidence score.
        
        Args:
            threshold: Confidence threshold
            
        Returns:
            Filtered match data
        """
        mask = self.confidence >= threshold
        return MatchData(
            image_pair=self.image_pair,
            matches=self.matches[mask],
            confidence=self.confidence[mask],
            match_type=self.match_type
        )


class FeatureMatcher:
    """Base class for feature matchers."""
    
    def __init__(self, name: str):
        """Initialize feature matcher.
        
        Args:
            name: Name of the feature matcher
        """
        self.name = name
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        """Match features between two images.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            Match data
        """
        raise NotImplementedError("Subclasses must implement match")
    
    def match_batch(self, features_list: List[FeatureData], pairs: List[Tuple[int, int]]) -> List[MatchData]:
        """Match features between multiple image pairs.
        
        Args:
            features_list: List of feature data for each image
            pairs: List of image pairs to match (i, j) where i < j
            
        Returns:
            List of match data
        """
        matches = []
        for i, j in tqdm(pairs, desc=f"Matching features using {self.name}"):
            if i >= len(features_list) or j >= len(features_list):
                logger.warning(f"Skipping invalid pair ({i}, {j})")
                continue
            
            # Ensure i < j
            if i > j:
                i, j = j, i
                
            match_data = self.match(features_list[i], features_list[j])
            matches.append(match_data)
        
        return matches


class MutualNNMatcher(FeatureMatcher):
    """Mutual Nearest Neighbor matcher."""
    
    def __init__(self, ratio_threshold: float = 0.8, cross_check: bool = True):
        """Initialize mutual nearest neighbor matcher.
        
        Args:
            ratio_threshold: Ratio test threshold (Lowe's ratio test)
            cross_check: Whether to use cross-checking
        """
        super().__init__("mutual_nn")
        self.ratio_threshold = ratio_threshold
        self.cross_check = cross_check
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        """Match features using mutual nearest neighbors with ratio test.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            Match data
        """
        # Extract descriptors
        desc1 = features1.descriptors
        desc2 = features2.descriptors
        
        # Check if we have descriptors
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            logger.warning("No descriptors to match")
            return MatchData(
                image_pair=(features1.image_id, features2.image_id),
                matches=np.zeros((0, 2), dtype=int),
                confidence=np.zeros(0, dtype=float),
                match_type=self.name
            )
        
        # Convert to appropriate data type if needed
        if isinstance(desc1, np.ndarray) and desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if isinstance(desc2, np.ndarray) and desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
        
        # Create BFMatcher
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        
        # Match descriptors (k=2 for ratio test)
        matches_1to2 = matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good_matches_1to2 = []
        for m, n in matches_1to2:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches_1to2.append(m)
        
        # If cross-check is enabled, do the same in reverse
        if self.cross_check:
            matches_2to1 = matcher.knnMatch(desc2, desc1, k=2)
            
            # Apply ratio test
            good_matches_2to1 = []
            for m, n in matches_2to1:
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches_2to1.append(m)
            
            # Find mutual matches
            mutual_matches = []
            confidence = []
            
            # Create reverse mapping for quick lookup
            reverse_map = {m.trainIdx: m.queryIdx for m in good_matches_2to1}
            
            for match in good_matches_1to2:
                query_idx = match.queryIdx
                train_idx = match.trainIdx
                
                # Check if there's a mutual match
                if train_idx in reverse_map and reverse_map[train_idx] == query_idx:
                    mutual_matches.append((query_idx, train_idx))
                    
                    # Compute confidence from distance
                    max_dist = 512.0  # Maximum possible distance for SIFT
                    conf = 1.0 - min(match.distance / max_dist, 1.0)
                    confidence.append(conf)
            
            matches = np.array(mutual_matches, dtype=int)
            confidence = np.array(confidence, dtype=float)
            
        else:
            # Without cross-check, just use the ratio test matches
            matches = np.array([(m.queryIdx, m.trainIdx) for m in good_matches_1to2], dtype=int)
            
            # Compute confidence scores
            max_dist = 512.0  # Maximum possible distance for SIFT
            confidence = np.array([1.0 - min(m.distance / max_dist, 1.0) for m in good_matches_1to2], dtype=float)
        
        return MatchData(
            image_pair=(features1.image_id, features2.image_id),
            matches=matches,
            confidence=confidence,
            match_type=self.name
        )


class FlannMatcher(FeatureMatcher):
    """FLANN-based feature matcher."""
    
    def __init__(self, ratio_threshold: float = 0.8, trees: int = 5, checks: int = 50):
        """Initialize FLANN matcher.
        
        Args:
            ratio_threshold: Ratio test threshold
            trees: Number of KD-trees for FLANN
            checks: Number of checks for FLANN
        """
        super().__init__("flann")
        self.ratio_threshold = ratio_threshold
        self.trees = trees
        self.checks = checks
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        """Match features using FLANN.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            Match data
        """
        # Extract descriptors
        desc1 = features1.descriptors
        desc2 = features2.descriptors
        
        # Check if we have descriptors
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            logger.warning("No descriptors to match")
            return MatchData(
                image_pair=(features1.image_id, features2.image_id),
                matches=np.zeros((0, 2), dtype=int),
                confidence=np.zeros(0, dtype=float),
                match_type=self.name
            )
        
        # Convert to appropriate data type if needed
        if isinstance(desc1, np.ndarray) and desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
        if isinstance(desc2, np.ndarray) and desc2.dtype != np.float32:
            desc2 = desc2.astype(np.float32)
        
        # Configure FLANN
        # For binary descriptors like ORB, use:
        if features1.feature_type in ["orb", "brief", "brisk"]:
            # For binary descriptors
            flann_params = dict(
                algorithm=6,  # FLANN_INDEX_LSH
                table_number=6,
                key_size=12,
                multi_probe_level=1
            )
        else:
            # For floating-point descriptors (SIFT, SURF, etc.)
            flann_params = dict(
                algorithm=1,  # FLANN_INDEX_KDTREE
                trees=self.trees
            )
            
        search_params = dict(checks=self.checks)
        
        # Create FLANN matcher
        flann = cv2.FlannBasedMatcher(flann_params, search_params)
        
        # Match descriptors (k=2 for ratio test)
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for i, (m, n) in enumerate(matches):
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        # Extract match indices and compute confidence
        match_indices = np.array([(m.queryIdx, m.trainIdx) for m in good_matches], dtype=int)
        
        # Compute confidence based on distance
        max_dist = 512.0  # Maximum possible distance
        confidence = np.array([1.0 - min(m.distance / max_dist, 1.0) for m in good_matches], dtype=float)
        
        return MatchData(
            image_pair=(features1.image_id, features2.image_id),
            matches=match_indices,
            confidence=confidence,
            match_type=self.name
        )


class SuperGlueMatcher(FeatureMatcher):
    """SuperGlue feature matcher based on learned features."""
    
    MODEL_URLS = {
        'superglue': 'https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_indoor.pth',
    }
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 device: Optional[str] = None):
        """Initialize SuperGlue matcher.
        
        Args:
            confidence_threshold: Confidence threshold for matches
            device: Device to use for inference ('cuda' or 'cpu')
        """
        super().__init__("superglue")
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load SuperGlue model."""
        try:
            # Try to import SuperGlue
            from models.superglue import SuperGlue
            
            # Create model
            config = {
                'descriptor_dim': 256,
                'weights': 'indoor',
                'keypoint_encoder': [32, 64, 128, 256],
                'GNN_layers': ['self', 'cross'] * 9,
                'sinkhorn_iterations': 100,
                'match_threshold': self.confidence_threshold,
            }
            
            model = SuperGlue(config)
            model.eval()
            
            # Download weights if needed
            weights_path = self._get_model_path()
            
            # Load weights
            weights = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(weights)
            
            # Move to device
            model = model.to(self.device)
            
            logger.info("SuperGlue model loaded successfully")
            
            return model
            
        except ImportError:
            # Fallback to minimal SuperGlue implementation
            logger.warning("SuperGlue package not found, using minimal implementation")
            
            return MinimalSuperGlue(
                descriptor_dim=256,
                confidence_threshold=self.confidence_threshold,
                device=self.device
            )
    
    def _get_model_path(self) -> str:
        """Get path to model weights, downloading if necessary."""
        # Create cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".recontext", "models")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine model filename
        model_url = self.MODEL_URLS['superglue']
        filename = os.path.basename(model_url)
        model_path = os.path.join(cache_dir, filename)
        
        # Download if needed
        if not os.path.exists(model_path):
            logger.info(f"Downloading SuperGlue model...")
            download_model(model_url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return model_path
    
    def match(self, features1: FeatureData, features2: FeatureData) -> MatchData:
        """Match features using SuperGlue.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            Match data
        """
        # Extract data
        kpts1 = features1.keypoints
        kpts2 = features2.keypoints
        desc1 = features1.descriptors
        desc2 = features2.descriptors
        scores1 = features1.scores
        scores2 = features2.scores
        
        # Prepare inputs for SuperGlue
        data = {
            'keypoints0': torch.from_numpy(kpts1).float().to(self.device),
            'keypoints1': torch.from_numpy(kpts2).float().to(self.device),
            'descriptors0': torch.from_numpy(desc1).float().to(self.device),
            'descriptors1': torch.from_numpy(desc2).float().to(self.device),
            'scores0': torch.from_numpy(scores1).float().to(self.device),
            'scores1': torch.from_numpy(scores2).float().to(self.device),
            'image_size0': torch.tensor(features1.image_size).to(self.device),
            'image_size1': torch.tensor(features2.image_size).to(self.device),
        }
        
        # Perform matching
        with torch.no_grad():
            pred = self.model(data)
        
        # Get matches
        indices0 = pred['indices0'].cpu().numpy()
        indices1 = pred['indices1'].cpu().numpy()
        matches = []
        confidence = []
        
        for i, j in enumerate(indices0):
            if j > -1:  # Valid match
                matches.append((i, indices1[i]))
                conf = pred['matching_scores0'][i].cpu().item()
                confidence.append(conf)
        
        # Convert to numpy arrays
        matches = np.array(matches, dtype=int)
        confidence = np.array(confidence, dtype=float)
        
        return MatchData(
            image_pair=(features1.image_id, features2.image_id),
            matches=matches,
            confidence=confidence,
            match_type=self.name
        )


class MinimalSuperGlue(torch.nn.Module):
    """Minimal SuperGlue implementation for feature matching."""
    
    def __init__(self, 
                 descriptor_dim: int = 256,
                 confidence_threshold: float = 0.5,
                 device: torch.device = None):
        """Initialize minimal SuperGlue model.
        
        Args:
            descriptor_dim: Descriptor dimension
            confidence_threshold: Confidence threshold for matches
            device: Device to use
        """
        super().__init__()
        
        self.descriptor_dim = descriptor_dim
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Create simple layers for keypoint encoding
        self.kp_encoder = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, descriptor_dim)
        )
        
        # GNN layers for descriptor refinement
        self.self_attn0 = torch.nn.MultiheadAttention(descriptor_dim, 4)
        self.self_attn1 = torch.nn.MultiheadAttention(descriptor_dim, 4)
        self.cross_attn0 = torch.nn.MultiheadAttention(descriptor_dim, 4)
        self.cross_attn1 = torch.nn.MultiheadAttention(descriptor_dim, 4)
        
        # Final MLP for score prediction
        self.final_proj = torch.nn.Sequential(
            torch.nn.Linear(descriptor_dim * 2, descriptor_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(descriptor_dim, 1)
        )
        
        # Move to device
        self.to(device)
    
    def forward(self, data: Dict) -> Dict:
        """Forward pass.
        
        Args:
            data: Dictionary with feature data
            
        Returns:
            Dictionary with match results
        """
        # Extract inputs
        kpts0 = data['keypoints0']
        kpts1 = data['keypoints1']
        desc0 = data['descriptors0']
        desc1 = data['descriptors1']
        
        # Normalize keypoints by image size
        size0 = data.get('image_size0', torch.tensor([1.0, 1.0]).to(self.device))
        size1 = data.get('image_size1', torch.tensor([1.0, 1.0]).to(self.device))
        
        norm_kpts0 = kpts0 / size0.view(1, 2)
        norm_kpts1 = kpts1 / size1.view(1, 2)
        
        # Encode keypoints
        kpts_enc0 = self.kp_encoder(norm_kpts0)
        kpts_enc1 = self.kp_encoder(norm_kpts1)
        
        # Combine with descriptors
        desc0 = desc0 + kpts_enc0
        desc1 = desc1 + kpts_enc1
        
        # Self and cross attention (simplified GNN)
        desc0_t = desc0.transpose(0, 1)
        desc1_t = desc1.transpose(0, 1)
        
        # Self attention
        desc0_t, _ = self.self_attn0(desc0_t, desc0_t, desc0_t)
        desc1_t, _ = self.self_attn1(desc1_t, desc1_t, desc1_t)
        
        # Cross attention
        desc0_t, _ = self.cross_attn0(desc0_t, desc1_t, desc1_t)
        desc1_t, _ = self.cross_attn1(desc1_t, desc0_t, desc0_t)
        
        desc0 = desc0_t.transpose(0, 1)
        desc1 = desc1_t.transpose(0, 1)
        
        # Compute scores (simplified Sinkhorn)
        scores = torch.matmul(desc0, desc1.transpose(0, 1))
        scores = scores / torch.sqrt(torch.tensor(self.descriptor_dim, dtype=torch.float32, device=self.device))
        
        # Get matches
        max_scores0, indices0 = torch.max(scores, dim=1)
        max_scores1, indices1 = torch.max(scores, dim=0)
        
        # Mutual nearest neighbors
        mutual_matches = torch.arange(len(kpts0), device=self.device) == indices1[indices0]
        indices0[~mutual_matches] = -1
        matching_scores0 = max_scores0 * mutual_matches.float()
        
        # Filter by confidence threshold
        valid_matches = matching_scores0 > self.confidence_threshold
        indices0[~valid_matches] = -1
        
        return {
            'indices0': indices0,
            'indices1': indices0.clone().to(self.device),  # simplified, in reality more complex
            'matching_scores0': matching_scores0
        }


def create_feature_matcher(matcher_type: str, **kwargs) -> FeatureMatcher:
    """Factory function for creating feature matchers.
    
    Args:
        matcher_type: Type of feature matcher ('mutual_nn', 'flann', 'superglue')
        **kwargs: Additional parameters for the matcher
        
    Returns:
        Feature matcher instance
    """
    if matcher_type == 'mutual_nn':
        return MutualNNMatcher(**kwargs)
    elif matcher_type == 'flann':
        return FlannMatcher(**kwargs)
    elif matcher_type == 'superglue':
        return SuperGlueMatcher(**kwargs)
    else:
        logger.warning(f"Unknown matcher type: {matcher_type}, using mutual_nn")
        return MutualNNMatcher(**kwargs)


def match_features(features: List[FeatureData], 
                  matcher_type: str = 'mutual_nn',
                  pairs: Optional[List[Tuple[int, int]]] = None,
                  **kwargs) -> List[MatchData]:
    """Match features between images.
    
    Args:
        features: List of feature data for each image
        matcher_type: Type of matcher to use
        pairs: List of image pairs to match (i, j) where i < j
        **kwargs: Additional parameters for the matcher
        
    Returns:
        List of match data
    """
    # Create matcher
    matcher = create_feature_matcher(matcher_type, **kwargs)
    
    # Set image IDs if not already set
    for i, feature in enumerate(features):
        if not hasattr(feature, 'image_id'):
            feature.image_id = i
    
    # Generate all pairs if not provided
    if pairs is None:
        pairs = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                pairs.append((i, j))
    
    # Match features
    logger.info(f"Matching features using {matcher_type} for {len(pairs)} image pairs")
    matches = matcher.match_batch(features, pairs)
    
    # Log statistics
    total_matches = sum(m.num_matches for m in matches)
    avg_matches = total_matches / len(matches) if matches else 0
    logger.info(f"Found {total_matches} total matches ({avg_matches:.1f} per pair)")
    
    return matches


def visualize_matches(image1: np.ndarray,
                     image2: np.ndarray,
                     features1: FeatureData,
                     features2: FeatureData,
                     matches: MatchData) -> np.ndarray:
    """Visualize feature matches between two images.
    
    Args:
        image1: First image
        image2: Second image
        features1: Features from first image
        features2: Features from second image
        matches: Match data
        
    Returns:
        Visualization image with matches
    """
    # Convert features to OpenCV keypoints
    kp1 = features1.to_cv_keypoints()
    kp2 = features2.to_cv_keypoints()
    
    # Convert matches to OpenCV DMatch objects
    dm = [cv2.DMatch(int(i), int(j), 0) for i, j in matches.matches]
    
    # Draw matches
    vis_img = cv2.drawMatches(image1, kp1, image2, kp2, dm, None,
                             matchesMask=None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return vis_img


def filter_matches_geometric(features1: FeatureData,
                           features2: FeatureData,
                           matches: MatchData,
                           method: str = 'fundamental',
                           ransac_threshold: float = 4.0) -> MatchData:
    """Filter matches using geometric verification.
    
    Args:
        features1: Features from first image
        features2: Features from second image
        matches: Match data
        method: Verification method ('fundamental', 'homography', 'essential')
        ransac_threshold: RANSAC threshold
        
    Returns:
        Filtered match data
    """
    if matches.num_matches < 8:
        logger.warning("Not enough matches for geometric verification")
        return matches
    
    # Extract matched points
    pts1 = features1.keypoints[matches.matches[:, 0]]
    pts2 = features2.keypoints[matches.matches[:, 1]]
    
    # Apply geometric verification
    if method == 'fundamental':
        # Estimate fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_threshold)
        if F is None or F.shape != (3, 3):
            logger.warning("Failed to estimate fundamental matrix")
            return matches
            
    elif method == 'homography':
        # Estimate homography
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_threshold)
        if H is None:
            logger.warning("Failed to estimate homography")
            return matches
            
    elif method == 'essential':
        # For essential matrix, we need camera intrinsics
        # This is a simplification - in practice, we would get real intrinsics
        w1, h1 = features1.image_size
        w2, h2 = features2.image_size
        
        K1 = np.array([
            [max(w1, h1), 0, w1 / 2],
            [0, max(w1, h1), h1 / 2],
            [0, 0, 1]
        ])
        
        K2 = np.array([
            [max(w2, h2), 0, w2 / 2],
            [0, max(w2, h2), h2 / 2],
            [0, 0, 1]
        ])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, threshold=ransac_threshold)
        if E is None or E.shape != (3, 3):
            logger.warning("Failed to estimate essential matrix")
            return matches
            
    else:
        logger.warning(f"Unknown geometric verification method: {method}")
        return matches
    
    # Convert mask to boolean array
    if mask is not None:
        mask = mask.ravel().astype(bool)
        
        # Filter matches
        filtered_matches = matches.matches[mask]
        filtered_confidence = matches.confidence[mask]
        
        filtered_match_data = MatchData(
            image_pair=matches.image_pair,
            matches=filtered_matches,
            confidence=filtered_confidence,
            match_type=matches.match_type + "_" + method
        )
        
        logger.info(f"Geometric verification ({method}): {filtered_match_data.num_matches}/{matches.num_matches} matches retained")
        
        return filtered_match_data
    else:
        logger.warning("Geometric verification produced no mask")
        return matches


def main():
    """Example usage of feature matching."""
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Feature matching example")
    parser.add_argument("--image1", required=True, help="First input image")
    parser.add_argument("--image2", required=True, help="Second input image")
    parser.add_argument("--feature_type", choices=["sift", "orb", "superpoint"],
                       default="sift", help="Feature type")
    parser.add_argument("--matcher_type", choices=["mutual_nn", "flann", "superglue"],
                       default="mutual_nn", help="Matcher type")
    parser.add_argument("--output", default="matches.jpg",
                       help="Output visualization")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Import feature extraction
    from recontext.core.feature_extraction import extract_features
    
    # Load images
    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)
    
    if image1 is None or image2 is None:
        logger.error(f"Failed to load images: {args.image1}, {args.image2}")
        return
    
    # Extract features
    features = extract_features(
        [image1, image2], 
        feature_type=args.feature_type
    )
    
    # Match features
    matches = match_features(
        features, 
        matcher_type=args.matcher_type,
        pairs=[(0, 1)]
    )[0]
    
    logger.info(f"Found {matches.num_matches} matches between images")
    
    # Apply geometric verification
    filtered_matches = filter_matches_geometric(
        features[0], features[1], matches, method='fundamental'
    )
    
    logger.info(f"After geometric verification: {filtered_matches.num_matches} matches")
    
    # Visualize matches
    vis_img = visualize_matches(image1, image2, features[0], features[1], filtered_matches)
    
    # Save visualization
    cv2.imwrite(args.output, vis_img)
    logger.info(f"Visualization saved to {args.output}")
    
    # Show visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Feature Matches ({args.feature_type} + {args.matcher_type})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()