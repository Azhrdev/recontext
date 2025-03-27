#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instance Segmentation module using Mask2Former and CLIP for zero-shot recognition.

This integrates state-of-the-art segmentation with vision-language models to provide
robust instance and semantic segmentation with the ability to recognize novel objects.

Author: Sarah Lin
Created: 2023-01-15
Last modified: 2024-03-07
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
import cv2
from tqdm import tqdm
from dataclasses import dataclass
from PIL import Image

# Try to import Detectron2 and Mask2Former
try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import MetadataCatalog
    from detectron2.engine.defaults import DefaultPredictor
    has_detectron2 = True
except ImportError:
    has_detectron2 = False
    logging.warning("Detectron2 not found. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")

# Try to import Mask2Former
try:
    from mask2former import add_maskformer2_config
    has_mask2former = True
except ImportError:
    has_mask2former = False
    logging.warning("Mask2Former not found. Install with: pip install 'git+https://github.com/facebookresearch/Mask2Former.git'")

# Try to import CLIP
try:
    import clip
    has_clip = True
except ImportError:
    has_clip = False
    logging.warning("CLIP not found. Install with: pip install 'git+https://github.com/openai/CLIP.git'")

from recontext.semantics.clip_embeddings import CLIPEmbedder
from recontext.semantics.label_manager import LabelManager
from recontext.utils.io_utils import download_model

logger = logging.getLogger(__name__)

@dataclass
class InstanceData:
    """Data structure for instance segmentation results."""
    mask: np.ndarray  # Binary mask
    class_id: int  # Class ID
    class_name: str  # Class name
    score: float  # Confidence score
    bbox: Tuple[int, int, int, int]  # Bounding box (x1, y1, x2, y2)
    color: Tuple[int, int, int]  # RGB color for visualization
    embedding: Optional[np.ndarray] = None  # CLIP embedding (if available)


class InstanceSegmentor:
    """Instance segmentation using Mask2Former with CLIP integration for zero-shot capabilities."""
    
    MODEL_URLS = {
        'mask2former_coco': 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl',
        'mask2former_ade20k': 'https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k/model_final_e0c58e.pkl',
        'swin_large': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
    }
    
    def __init__(self, 
                 model_type: str = 'mask2former_coco',
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 enable_clip: bool = True):
        """Initialize instance segmentation model.
        
        Args:
            model_type: Model type to use (mask2former_coco or mask2former_ade20k)
            device: Device to use (cuda or cpu)
            confidence_threshold: Confidence threshold for detections
            enable_clip: Enable CLIP for zero-shot recognition
        """
        if not has_detectron2 or not has_mask2former:
            raise ImportError("Detectron2 and Mask2Former are required for instance segmentation")
        
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.enable_clip = enable_clip and has_clip
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_mask2former()
        
        if self.enable_clip:
            logger.info("Initializing CLIP for zero-shot recognition")
            self.clip_embedder = CLIPEmbedder(device=self.device)
            self.label_manager = LabelManager()
        
    def _init_mask2former(self):
        """Initialize Mask2Former model."""
        logger.info(f"Initializing Mask2Former model: {self.model_type}")
        
        # Create config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        
        # Download model weights if needed
        model_path = self._get_model_path()
        
        # Set configuration based on model type
        if self.model_type == 'mask2former_coco':
            cfg.merge_from_file("configs/coco/instance-segmentation/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
            cfg.MODEL.WEIGHTS = model_path
            num_classes = 80  # COCO dataset has 80 classes
        elif self.model_type == 'mask2former_ade20k':
            cfg.merge_from_file("configs/ade20k/semantic-segmentation/maskformer2_swin_large_IN21k_384_bs16_160k.yaml")
            cfg.MODEL.WEIGHTS = model_path
            num_classes = 150  # ADE20K dataset has 150 classes
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Common configuration
        cfg.MODEL.DEVICE = 'cuda' if self.device.type == 'cuda' else 'cpu'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        
        # Create predictor
        try:
            self.predictor = DefaultPredictor(cfg)
            logger.info("Mask2Former model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mask2Former: {e}")
            raise
        
        # Get metadata for class names
        if self.model_type == 'mask2former_coco':
            self.metadata = MetadataCatalog.get("coco_2017_val")
        else:
            self.metadata = MetadataCatalog.get("ade20k_sem_seg_val")
            
        # Get class names
        self.class_names = self.metadata.thing_classes if hasattr(self.metadata, 'thing_classes') else self.metadata.stuff_classes
        logger.info(f"Loaded {len(self.class_names)} class names")
    
    def _get_model_path(self) -> str:
        """Get path to model weights, downloading if necessary."""
        # Create cache directory
        cache_dir = os.path.join(os.path.expanduser("~"), ".recontext", "models")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Determine model filename
        model_name = self.model_type
        if model_name not in self.MODEL_URLS:
            raise ValueError(f"Unknown model type: {model_name}")
        
        model_url = self.MODEL_URLS[model_name]
        filename = os.path.basename(model_url)
        model_path = os.path.join(cache_dir, filename)
        
        # Download if needed
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_name} model...")
            download_model(model_url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        
        return model_path
    
    def process_image(self, image: Union[np.ndarray, str]) -> List[InstanceData]:
        """Process a single image for instance segmentation.
        
        Args:
            image: Input image (numpy array or path to image)
            
        Returns:
            List of InstanceData objects
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with Mask2Former
        with torch.no_grad():
            outputs = self.predictor(image)
        
        # Debug info about outputs
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Prediction output keys: {outputs['instances'].get_fields().keys()}")
            
        # Extract results
        instances = []
        
        # Get output tensors
        if "instances" not in outputs:
            logger.warning("No instances detected in image")
            return []
            
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy() if outputs["instances"].has("pred_boxes") else None
        scores = outputs["instances"].scores.cpu().numpy() if outputs["instances"].has("scores") else None
        pred_classes = outputs["instances"].pred_classes.cpu().numpy() if outputs["instances"].has("pred_classes") else None
        pred_masks = outputs["instances"].pred_masks.cpu().numpy() if outputs["instances"].has("pred_masks") else None
        
        if pred_masks is None or len(pred_masks) == 0:
            logger.warning("No masks detected in image")
            return []
        
        # Process each instance
        for i in range(len(pred_masks)):
            if scores is not None and scores[i] < self.confidence_threshold:
                continue
                
            # Get mask
            mask = pred_masks[i].astype(np.uint8)
            
            # Get class info
            class_id = int(pred_classes[i]) if pred_classes is not None else -1
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else "unknown"
            score = float(scores[i]) if scores is not None else 1.0
            
            # Get bounding box
            if pred_boxes is not None:
                bbox = tuple(map(int, pred_boxes[i]))
            else:
                # Compute bounding box from mask
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x1, x2 = np.min(x_indices), np.max(x_indices)
                    y1, y2 = np.min(y_indices), np.max(y_indices)
                    bbox = (x1, y1, x2, y2)
                else:
                    bbox = (0, 0, 0, 0)
            
            # Generate color for visualization (deterministic based on class_id)
            color = tuple(map(int, np.random.RandomState(class_id).randint(0, 255, 3)))
            
            # Create instance data
            instance = InstanceData(
                mask=mask,
                class_id=class_id,
                class_name=class_name,
                score=score,
                bbox=bbox,
                color=color
            )
            
            instances.append(instance)
        
        # Sort by score (descending)
        instances = sorted(instances, key=lambda x: x.score, reverse=True)
        
        # Apply CLIP for zero-shot recognition if enabled
        if self.enable_clip and instances:
            # TODO: this is a bit inefficient, we should batch process all instances
            for i, instance in enumerate(instances):
                # Extract instance patch from image using mask
                y_indices, x_indices = np.where(instance.mask > 0)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                    
                x1, x2 = max(0, np.min(x_indices)), min(image.shape[1]-1, np.max(x_indices))
                y1, y2 = max(0, np.min(y_indices)), min(image.shape[0]-1, np.max(y_indices))
                
                # Ensure we have a valid patch
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                patch = image[y1:y2, x1:x2]
                
                # Skip tiny patches
                if patch.shape[0] < 10 or patch.shape[1] < 10:
                    continue
                
                # Convert to PIL for CLIP
                patch_pil = Image.fromarray(patch)
                
                # Get CLIP embedding
                embedding = self.clip_embedder.embed_image(patch_pil)
                instance.embedding = embedding
                
                # Get zero-shot class prediction if we're uncertain about the class
                if instance.score < 0.7 and embedding is not None:
                    # Get potential class names from the label manager
                    potentials = self.label_manager.get_potential_classes(instance.class_name)
                    
                    if potentials:
                        # Get text embeddings for potential classes
                        text_embeddings = self.clip_embedder.embed_text(potentials)
                        
                        # Compute similarities
                        similarities = self.clip_embedder.compute_similarity(embedding, text_embeddings)
                        
                        # Get best match
                        best_idx = np.argmax(similarities)
                        best_class = potentials[best_idx]
                        best_score = similarities[best_idx]
                        
                        # Update class if CLIP is confident
                        if best_score > 0.25:  # Threshold for CLIP confidence
                            logger.debug(f"CLIP updated class from {instance.class_name} to {best_class} with score {best_score:.3f}")
                            instance.class_name = best_class
        
        return instances
    
    def process_batch(self, images: List[Union[np.ndarray, str]]) -> List[List[InstanceData]]:
        """Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of lists of InstanceData objects, one list per image
        """
        logger.info(f"Processing batch of {len(images)} images")
        results = []
        
        for i, image in enumerate(tqdm(images, desc="Segmenting images")):
            image_results = self.process_image(image)
            results.append(image_results)
            
        return results
    
    def visualize_results(self, image: np.ndarray, instances: List[InstanceData], 
                          alpha: float = 0.5) -> np.ndarray:
        """Visualize segmentation results on image.
        
        Args:
            image: Input image
            instances: List of InstanceData objects
            alpha: Transparency for mask overlay
            
        Returns:
            Visualization image with instance masks and labels
        """
        vis_image = image.copy()
        
        # Sort instances by area (smallest to largest) for better visualization
        sorted_instances = sorted(instances, 
                                 key=lambda x: np.sum(x.mask),
                                 reverse=False)
        
        mask_overlay = np.zeros_like(image)
        
        # Draw instance masks
        for instance in sorted_instances:
            mask = instance.mask
            color = instance.color
            
            # Apply color to mask regions
            for c in range(3):
                mask_overlay[:, :, c] = np.where(mask == 1, 
                                               color[c], 
                                               mask_overlay[:, :, c])
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, alpha, 0)
        
        # Draw bounding boxes and labels
        for instance in sorted_instances:
            x1, y1, x2, y2 = instance.bbox
            color = instance.color
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            text = f"{instance.class_name}: {instance.score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw text background
            cv2.rectangle(vis_image, 
                         (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), 
                         color, -1)
            
            # Draw text
            cv2.putText(vis_image, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image


# Main function for standalone usage
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run instance segmentation")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output visualization path")
    parser.add_argument("--model", default="mask2former_coco", 
                       choices=["mask2former_coco", "mask2former_ade20k"],
                       help="Model type to use")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Confidence threshold")
    parser.add_argument("--clip", action="store_true",
                       help="Enable CLIP for zero-shot recognition")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize segmentor
    segmentor = InstanceSegmentor(
        model_type=args.model,
        confidence_threshold=args.threshold,
        enable_clip=args.clip
    )
    
    # Process image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    instances = segmentor.process_image(image)
    logger.info(f"Detected {len(instances)} instances")
    
    # Visualize results
    vis_image = segmentor.visualize_results(image, instances)
    
    # Save visualization
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, vis_image_bgr)
    logger.info(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()