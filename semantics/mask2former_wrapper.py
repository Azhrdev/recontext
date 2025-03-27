#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mask2Former wrapper for semantic segmentation.

This module provides a high-level interface to use Mask2Former models
for instance and semantic segmentation tasks.

Author: Sarah Lin
Date: 2024-01-20
Last modified: 2024-03-05
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import cv2
from PIL import Image

from recontext.semantics.label_manager import LabelManager
from recontext.utils.io_utils import download_model

# Try to import Detectron2 dependencies
try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.engine.defaults import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import MetadataCatalog
    HAS_DETECTRON2 = True
except ImportError:
    HAS_DETECTRON2 = False
    logging.warning("Detectron2 not available. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")

# Try to import Mask2Former
try:
    from mask2former import add_maskformer2_config
    HAS_MASK2FORMER = True
except ImportError:
    HAS_MASK2FORMER = False
    logging.warning("Mask2Former not available. Install with: pip install 'git+https://github.com/facebookresearch/Mask2Former.git'")

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Container for Mask2Former segmentation results."""
    masks: np.ndarray  # N x H x W binary masks
    class_ids: np.ndarray  # N class ids
    class_names: List[str]  # N class names
    scores: np.ndarray  # N confidence scores
    boxes: Optional[np.ndarray] = None  # N x 4 bounding boxes (x1, y1, x2, y2) if available


class Mask2FormerWrapper:
    """Wrapper for Mask2Former models."""
    
    MODEL_CONFIGS = {
        'coco_instance': {
            'config': "configs/coco/instance-segmentation/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
            'weights': "mask2former_coco.pkl",
            'url': "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl",
            'dataset': 'coco',
            'task': 'instance'
        },
        'coco_panoptic': {
            'config': "configs/coco/panoptic-segmentation/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
            'weights': "mask2former_coco_panoptic.pkl",
            'url': "https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl",
            'dataset': 'coco',
            'task': 'panoptic'
        },
        'ade20k_semantic': {
            'config': "configs/ade20k/semantic-segmentation/maskformer2_swin_large_IN21k_384_bs16_160k.yaml",
            'weights': "mask2former_ade20k.pkl",
            'url': "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_e5f453.pkl",
            'dataset': 'ade20k',
            'task': 'semantic'
        },
    }
    
    def __init__(self, model_type: str = 'coco_instance', 
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """Initialize Mask2Former wrapper.
        
        Args:
            model_type: Model type (coco_instance, coco_panoptic, ade20k_semantic)
            device: Device to use (cuda or cpu)
            confidence_threshold: Confidence threshold for predictions
        """
        if not HAS_DETECTRON2 or not HAS_MASK2FORMER:
            raise ImportError("Detectron2 and Mask2Former are required for this module")
        
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        device_str = 'cuda' if self.device.type == 'cuda' else 'cpu'
        logger.info(f"Using device: {device_str}")
        
        # Initialize label manager
        dataset = self.MODEL_CONFIGS[model_type]['dataset']
        self.label_manager = LabelManager([dataset])
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Mask2Former model."""
        if self.model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_config = self.MODEL_CONFIGS[self.model_type]
        logger.info(f"Initializing Mask2Former model: {self.model_type}")
        
        # Ensure model weights are available
        model_path = self._get_weights_path(model_config)
        
        # Configure model
        try:
            cfg = get_cfg()
            add_deeplab_config(cfg)
            add_maskformer2_config(cfg)
            
            # Load config file - might need adjustments based on your setup
            try:
                cfg.merge_from_file(model_config['config'])
            except Exception as e:
                logger.warning(f"Could not load config file directly: {e}")
                # Handle the case where the config file isn't in the expected location
                pass
            
            # Set weights path
            cfg.MODEL.WEIGHTS = model_path
            
            # Set device
            cfg.MODEL.DEVICE = 'cuda' if self.device.type == 'cuda' else 'cpu'
            
            # Set confidence threshold
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
            
            # Create predictor
            self.predictor = DefaultPredictor(cfg)
            
            # Get metadata for class names
            if model_config['dataset'] == 'coco':
                if model_config['task'] == 'instance':
                    self.metadata = MetadataCatalog.get("coco_2017_val")
                elif model_config['task'] == 'panoptic':
                    self.metadata = MetadataCatalog.get("coco_2017_val_panoptic")
                else:
                    self.metadata = MetadataCatalog.get("coco_2017_val")
            elif model_config['dataset'] == 'ade20k':
                self.metadata = MetadataCatalog.get("ade20k_sem_seg_val")
            else:
                # Fallback
                self.metadata = None
                
            logger.info("Mask2Former model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Mask2Former model: {e}")
            raise
    
    def _get_weights_path(self, model_config: Dict) -> str:
        """Get path to model weights, downloading if necessary.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Path to model weights file
        """
        # Create cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".recontext", "models")
        os.makedirs(cache_dir, exist_ok=True)
        
        weights_file = model_config['weights']
        weights_path = os.path.join(cache_dir, weights_file)
        
        # Download if needed
        if not os.path.exists(weights_path):
            logger.info(f"Downloading {self.model_type} model...")
            download_model(model_config['url'], weights_path)
            logger.info(f"Model downloaded to {weights_path}")
            
        return weights_path
    
    def predict(self, image: Union[np.ndarray, str]) -> SegmentationResult:
        """Run segmentation on input image.
        
        Args:
            image: Input image or path to image
            
        Returns:
            Segmentation results
        """
        # Load image if path is provided
        if isinstance(image, str):
            try:
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Failed to load image {image}: {e}")
                raise
        
        # Ensure image is RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Run prediction
        with torch.no_grad():
            outputs = self.predictor(image)
        
        # Extract results
        if self.MODEL_CONFIGS[self.model_type]['task'] == 'instance':
            return self._process_instance_results(outputs, image.shape[:2])
        elif self.MODEL_CONFIGS[self.model_type]['task'] == 'semantic':
            return self._process_semantic_results(outputs, image.shape[:2])
        elif self.MODEL_CONFIGS[self.model_type]['task'] == 'panoptic':
            return self._process_panoptic_results(outputs, image.shape[:2])
        else:
            raise ValueError(f"Unsupported task: {self.MODEL_CONFIGS[self.model_type]['task']}")
    
    def _process_instance_results(self, outputs: Dict, image_shape: Tuple[int, int]) -> SegmentationResult:
        """Process instance segmentation results.
        
        Args:
            outputs: Model outputs
            image_shape: (height, width) of input image
            
        Returns:
            Segmentation results
        """
        # Check if instances are available
        if "instances" not in outputs:
            return SegmentationResult(
                masks=np.zeros((0, image_shape[0], image_shape[1]), dtype=bool),
                class_ids=np.zeros(0, dtype=int),
                class_names=[],
                scores=np.zeros(0, dtype=float),
                boxes=np.zeros((0, 4), dtype=float)
            )
        
        instances = outputs["instances"]
        
        # Get predictions
        if instances.has("pred_masks"):
            masks = instances.pred_masks.cpu().numpy()
        else:
            masks = np.zeros((0, image_shape[0], image_shape[1]), dtype=bool)
            
        if instances.has("pred_classes"):
            class_ids = instances.pred_classes.cpu().numpy()
        else:
            class_ids = np.zeros(len(masks), dtype=int)
            
        if instances.has("scores"):
            scores = instances.scores.cpu().numpy()
        else:
            scores = np.ones(len(masks), dtype=float)
            
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.cpu().numpy()
        else:
            boxes = None
        
        # Filter by confidence
        if len(scores) > 0:
            keep = scores >= self.confidence_threshold
            masks = masks[keep]
            class_ids = class_ids[keep]
            scores = scores[keep]
            if boxes is not None:
                boxes = boxes[keep]
        
        # Get class names
        if self.metadata and hasattr(self.metadata, "thing_classes"):
            thing_classes = self.metadata.thing_classes
            class_names = [thing_classes[class_id] if 0 <= class_id < len(thing_classes) else "unknown" 
                          for class_id in class_ids]
        else:
            class_names = [f"class_{class_id}" for class_id in class_ids]
        
        return SegmentationResult(
            masks=masks,
            class_ids=class_ids,
            class_names=class_names,
            scores=scores,
            boxes=boxes
        )
    
    def _process_semantic_results(self, outputs: Dict, image_shape: Tuple[int, int]) -> SegmentationResult:
        """Process semantic segmentation results.
        
        Args:
            outputs: Model outputs
            image_shape: (height, width) of input image
            
        Returns:
            Segmentation results
        """
        # Check if sem_seg is available
        if "sem_seg" not in outputs:
            return SegmentationResult(
                masks=np.zeros((0, image_shape[0], image_shape[1]), dtype=bool),
                class_ids=np.zeros(0, dtype=int),
                class_names=[],
                scores=np.zeros(0, dtype=float)
            )
        
        # Get semantic segmentation
        sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()
        
        # Find unique classes (excluding background)
        unique_ids = np.unique(sem_seg)
        if 0 in unique_ids:  # Remove background
            unique_ids = unique_ids[unique_ids != 0]
        
        # Create mask for each class
        masks = []
        class_ids = []
        scores = []
        
        for class_id in unique_ids:
            mask = (sem_seg == class_id)
            masks.append(mask)
            class_ids.append(class_id)
            scores.append(1.0)  # No confidence score for semantic segmentation
        
        # Get class names
        if self.metadata and hasattr(self.metadata, "stuff_classes"):
            stuff_classes = self.metadata.stuff_classes
            class_names = [stuff_classes[class_id] if 0 <= class_id < len(stuff_classes) else "unknown" 
                          for class_id in class_ids]
        else:
            class_names = [f"class_{class_id}" for class_id in class_ids]
        
        return SegmentationResult(
            masks=np.array(masks) if masks else np.zeros((0, image_shape[0], image_shape[1]), dtype=bool),
            class_ids=np.array(class_ids),
            class_names=class_names,
            scores=np.array(scores)
        )
    
    def _process_panoptic_results(self, outputs: Dict, image_shape: Tuple[int, int]) -> SegmentationResult:
        """Process panoptic segmentation results.
        
        Args:
            outputs: Model outputs
            image_shape: (height, width) of input image
            
        Returns:
            Segmentation results
        """
        # Check if panoptic_seg is available
        if "panoptic_seg" not in outputs:
            return SegmentationResult(
                masks=np.zeros((0, image_shape[0], image_shape[1]), dtype=bool),
                class_ids=np.zeros(0, dtype=int),
                class_names=[],
                scores=np.zeros(0, dtype=float)
            )
        
        # Get panoptic segmentation
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        panoptic_seg = panoptic_seg.cpu().numpy()
        
        # Process each segment
        masks = []
        class_ids = []
        scores = []
        
        for segment in segments_info:
            # Get segment ID and class
            segment_id = segment["id"]
            category_id = segment["category_id"]
            isthing = segment.get("isthing", True)
            score = segment.get("score", 1.0)
            
            # Skip segments with low confidence
            if score < self.confidence_threshold:
                continue
            
            # Create mask for this segment
            mask = (panoptic_seg == segment_id)
            masks.append(mask)
            class_ids.append(category_id)
            scores.append(score)
        
        # Get class names
        if self.metadata:
            if hasattr(self.metadata, "thing_classes") and hasattr(self.metadata, "stuff_classes"):
                thing_classes = self.metadata.thing_classes
                stuff_classes = self.metadata.stuff_classes
                
                class_names = []
                for class_id in class_ids:
                    if class_id < len(thing_classes):
                        class_names.append(thing_classes[class_id])
                    else:
                        # Adjust index for stuff classes
                        stuff_id = class_id - len(thing_classes)
                        if stuff_id < len(stuff_classes):
                            class_names.append(stuff_classes[stuff_id])
                        else:
                            class_names.append(f"class_{class_id}")
            else:
                class_names = [f"class_{class_id}" for class_id in class_ids]
        else:
            class_names = [f"class_{class_id}" for class_id in class_ids]
        
        return SegmentationResult(
            masks=np.array(masks) if masks else np.zeros((0, image_shape[0], image_shape[1]), dtype=bool),
            class_ids=np.array(class_ids),
            class_names=class_names,
            scores=np.array(scores)
        )
    
    def visualize(self, image: np.ndarray, result: SegmentationResult, 
                 alpha: float = 0.5, show_label: bool = True) -> np.ndarray:
        """Visualize segmentation results on image.
        
        Args:
            image: Input image
            result: Segmentation results
            alpha: Transparency for mask overlay
            show_label: Whether to show labels
            
        Returns:
            Visualization image
        """
        if len(result.masks) == 0:
            return image.copy()
            
        # Create copy of image
        vis_image = image.copy()
        
        # Create random colors for each class (deterministic based on class ID)
        colors = []
        for class_id in result.class_ids:
            # Use label manager for consistent colors if possible
            try:
                color = self.label_manager.get_color(int(class_id))
            except:
                # Random but deterministic color
                np.random.seed(int(class_id))
                color = np.random.randint(0, 255, size=3).tolist()
                np.random.seed(None)  # Reset seed
            
            colors.append(color)
        
        # Apply masks with transparency
        colored_masks = np.zeros_like(image)
        
        # Sort by score (high to low) so higher confidence instances are on top
        sort_indices = np.argsort(-result.scores)
        
        for i in sort_indices:
            mask = result.masks[i]
            color = colors[i]
            
            # Apply color to mask regions
            for c in range(3):
                colored_masks[:, :, c] = np.where(mask, color[c], colored_masks[:, :, c])
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 1.0, colored_masks, alpha, 0)
        
        # Add labels if requested
        if show_label and result.boxes is not None:
            for i in range(len(result.class_names)):
                # Get bounding box
                x1, y1, x2, y2 = map(int, result.boxes[i])
                
                # Draw label background
                label = f"{result.class_names[i]}: {result.scores[i]:.2f}"
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), colors[i], -1)
                
                # Draw label text
                cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), colors[i], 2)
        
        return vis_image


def main():
    """Example usage for demonstration."""
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Mask2Former wrapper")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default="coco_instance", choices=["coco_instance", "ade20k_semantic"],
                        help="Model type")
    parser.add_argument("--output", default="output.jpg", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        model = Mask2FormerWrapper(
            model_type=args.model,
            confidence_threshold=args.threshold
        )
        
        # Load image
        image = cv2.imread(args.image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        result = model.predict(image)
        
        # Visualize results
        vis_image = model.visualize(image, result)
        
        # Save output
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, vis_image_bgr)
        
        logger.info(f"Results saved to {args.output}")
        logger.info(f"Found {len(result.class_ids)} instances:")
        
        for i, (class_name, score) in enumerate(zip(result.class_names, result.scores)):
            logger.info(f"  {class_name}: {score:.3f}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()