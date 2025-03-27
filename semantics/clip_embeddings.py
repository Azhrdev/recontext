#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLIP embeddings module for zero-shot recognition.

This module provides interfaces to OpenAI's CLIP models for generating embeddings
of images and text, enabling zero-shot recognition and classification.

Author: Sarah Lin
Date: 2024-02-03
Last modified: 2024-03-10
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image

logger = logging.getLogger(__name__)

# Try to import CLIP
try:
    import clip
    has_clip = True
except ImportError:
    has_clip = False
    logger.warning("CLIP not found. Install with: pip install git+https://github.com/openai/CLIP.git")


class CLIPEmbedder:
    """CLIP embedding generator for zero-shot recognition."""
    
    AVAILABLE_MODELS = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"]
    
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 device: Optional[str] = None,
                 normalize: bool = True):
        """Initialize CLIP embedder.
        
        Args:
            model_name: CLIP model name
            device: Device to use
            normalize: Whether to normalize embeddings
        """
        if not has_clip:
            raise ImportError("CLIP is required for CLIPEmbedder")
        
        self.model_name = model_name
        self.normalize = normalize
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Loaded CLIP model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def embed_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Generate embedding for an image.
        
        Args:
            image: Image to embed (path, PIL image, or numpy array)
            
        Returns:
            Image embedding
        """
        # Load and preprocess image
        if isinstance(image, str):
            # Load from path
            try:
                image = Image.open(image).convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
                return None
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL image
            image = Image.fromarray(np.uint8(image)).convert('RGB')
        elif not isinstance(image, Image.Image):
            logger.error(f"Unsupported image type: {type(image)}")
            return None
        
        # Preprocess image
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            
            if self.normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            embedding = image_features.cpu().numpy().flatten()
        
        return embedding
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embedding for text.
        
        Args:
            text: Text to embed (string or list of strings)
            
        Returns:
            Text embedding(s)
        """
        # Handle single text or list
        if isinstance(text, str):
            text = [text]
        
        # Encode text
        with torch.no_grad():
            text_tokens = clip.tokenize(text).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            
            if self.normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            embeddings = text_features.cpu().numpy()
            
            # If single text, return flattened embedding
            if len(text) == 1:
                return embeddings.flatten()
            else:
                return embeddings
    
    def compute_similarity(self, 
                          image_embedding: np.ndarray,
                          text_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embedding: Image embedding
            text_embeddings: Text embeddings
            
        Returns:
            Array of similarity scores
        """
        # Ensure embeddings are normalized
        if not self.normalize:
            image_norm = np.linalg.norm(image_embedding)
            if image_norm > 0:
                image_embedding = image_embedding / image_norm
            
            if len(text_embeddings.shape) == 2:
                # Multiple text embeddings
                text_norms = np.linalg.norm(text_embeddings, axis=1, keepdims=True)
                valid_mask = text_norms > 0
                text_embeddings = np.copy(text_embeddings)
                text_embeddings[valid_mask] = text_embeddings[valid_mask] / text_norms[valid_mask]
            else:
                # Single text embedding
                text_norm = np.linalg.norm(text_embeddings)
                if text_norm > 0:
                    text_embeddings = text_embeddings / text_norm
        
        # Compute similarities
        if len(text_embeddings.shape) == 2:
            # Multiple text embeddings
            similarities = np.dot(text_embeddings, image_embedding)
        else:
            # Single text embedding
            similarities = np.array([np.dot(text_embeddings, image_embedding)])
        
        return similarities
    
    def classify_image(self, 
                      image: Union[str, Image.Image, np.ndarray],
                      class_labels: List[str],
                      prompt_template: str = "a photo of a {}") -> Tuple[str, float]:
        """Classify image using zero-shot recognition.
        
        Args:
            image: Image to classify
            class_labels: List of class labels
            prompt_template: Template for class prompts
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Embed image
        image_embedding = self.embed_image(image)
        if image_embedding is None:
            return None, 0.0
        
        # Create prompts
        prompts = [prompt_template.format(label) for label in class_labels]
        
        # Embed text
        text_embeddings = self.embed_text(prompts)
        
        # Compute similarities
        similarities = self.compute_similarity(image_embedding, text_embeddings)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return class_labels[best_idx], float(best_score)
    
    def batch_embed_images(self, images: List[Union[str, Image.Image, np.ndarray]]) -> np.ndarray:
        """Generate embeddings for a batch of images.
        
        Args:
            images: List of images to embed
            
        Returns:
            Array of image embeddings
        """
        # Process each image
        embeddings = []
        for image in images:
            embedding = self.embed_image(image)
            if embedding is not None:
                embeddings.append(embedding)
        
        if not embeddings:
            return np.array([])
            
        return np.stack(embeddings)
    
    def batch_classify_images(self, 
                             images: List[Union[str, Image.Image, np.ndarray]],
                             class_labels: List[str],
                             prompt_template: str = "a photo of a {}") -> List[Tuple[str, float]]:
        """Classify a batch of images using zero-shot recognition.
        
        Args:
            images: List of images to classify
            class_labels: List of class labels
            prompt_template: Template for class prompts
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        # Create prompts
        prompts = [prompt_template.format(label) for label in class_labels]
        
        # Embed text (single batch)
        text_embeddings = self.embed_text(prompts)
        
        # Process each image
        results = []
        for image in images:
            # Embed image
            image_embedding = self.embed_image(image)
            if image_embedding is None:
                results.append((None, 0.0))
                continue
            
            # Compute similarities
            similarities = self.compute_similarity(image_embedding, text_embeddings)
            
            # Get best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            results.append((class_labels[best_idx], float(best_score)))
        
        return results


class ZeroShotClassifier:
    """Zero-shot image classifier using CLIP embeddings."""
    
    def __init__(self, 
                 class_labels: List[str],
                 model_name: str = "ViT-B/32",
                 device: Optional[str] = None,
                 prompt_template: str = "a photo of a {}"):
        """Initialize zero-shot classifier.
        
        Args:
            class_labels: List of class labels
            model_name: CLIP model name
            device: Device to use
            prompt_template: Template for class prompts
        """
        self.class_labels = class_labels
        self.prompt_template = prompt_template
        
        # Initialize CLIP embedder
        self.clip_embedder = CLIPEmbedder(model_name=model_name, device=device)
        
        # Generate class embeddings
        self.class_embeddings = self._generate_class_embeddings()
    
    def _generate_class_embeddings(self) -> np.ndarray:
        """Generate embeddings for class labels.
        
        Returns:
            Array of class embeddings
        """
        # Create prompts
        prompts = [self.prompt_template.format(label) for label in self.class_labels]
        
        # Embed text
        embeddings = self.clip_embedder.embed_text(prompts)
        
        return embeddings
    
    def classify(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[str, float]:
        """Classify image.
        
        Args:
            image: Image to classify
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Embed image
        image_embedding = self.clip_embedder.embed_image(image)
        if image_embedding is None:
            return None, 0.0
        
        # Compute similarities
        similarities = self.clip_embedder.compute_similarity(image_embedding, self.class_embeddings)
        
        # Get best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        return self.class_labels[best_idx], float(best_score)
    
    def batch_classify(self, 
                      images: List[Union[str, Image.Image, np.ndarray]]) -> List[Tuple[str, float]]:
        """Classify a batch of images.
        
        Args:
            images: List of images to classify
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        for image in images:
            predicted_class, confidence = self.classify(image)
            results.append((predicted_class, confidence))
        
        return results
    
    def get_class_similarities(self, 
                              image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """Get similarities between image and all classes.
        
        Args:
            image: Image to compare
            
        Returns:
            Dictionary mapping class labels to similarity scores
        """
        # Embed image
        image_embedding = self.clip_embedder.embed_image(image)
        if image_embedding is None:
            return {}
        
        # Compute similarities
        similarities = self.clip_embedder.compute_similarity(image_embedding, self.class_embeddings)
        
        # Create mapping
        result = {}
        for i, label in enumerate(self.class_labels):
            result[label] = float(similarities[i])
        
        return result
    
    def add_class(self, class_label: str) -> None:
        """Add a new class to the classifier.
        
        Args:
            class_label: New class label
        """
        if class_label in self.class_labels:
            logger.warning(f"Class '{class_label}' already exists")
            return
        
        # Add to list
        self.class_labels.append(class_label)
        
        # Generate embedding
        prompt = self.prompt_template.format(class_label)
        embedding = self.clip_embedder.embed_text(prompt)
        
        # Add to embeddings
        if len(self.class_embeddings.shape) == 1:
            # First embedding, reshape
            self.class_embeddings = np.stack([self.class_embeddings, embedding])
        else:
            # Add to existing embeddings
            self.class_embeddings = np.vstack([self.class_embeddings, embedding])


def prompt_engineering(class_labels: List[str], 
                      prompt_templates: Optional[List[str]] = None) -> List[str]:
    """Generate improved prompts for zero-shot classification.
    
    Args:
        class_labels: List of class labels
        prompt_templates: Optional list of prompt templates
        
    Returns:
        List of prompts
    """
    if prompt_templates is None:
        # Default templates
        prompt_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a close-up photo of a {}",
            "a photo of the {}"
        ]
    
    # Generate prompts for each class
    all_prompts = []
    for label in class_labels:
        prompts = [template.format(label) for template in prompt_templates]
        all_prompts.extend(prompts)
    
    return all_prompts


class EnsembleZeroShotClassifier:
    """Ensemble zero-shot classifier using multiple models."""
    
    def __init__(self, 
                 class_labels: List[str],
                 model_names: List[str] = ["ViT-B/32", "ViT-B/16", "RN50"],
                 device: Optional[str] = None,
                 prompt_templates: Optional[List[str]] = None):
        """Initialize ensemble classifier.
        
        Args:
            class_labels: List of class labels
            model_names: List of CLIP model names
            device: Device to use
            prompt_templates: Optional list of prompt templates
        """
        self.class_labels = class_labels
        
        # Create prompt templates
        if prompt_templates is None:
            self.prompt_templates = [
                "a photo of a {}",
                "a picture of a {}",
                "an image of a {}"
            ]
        else:
            self.prompt_templates = prompt_templates
        
        # Initialize CLIP embedders
        self.embedders = []
        for model_name in model_names:
            try:
                embedder = CLIPEmbedder(model_name=model_name, device=device)
                self.embedders.append(embedder)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
    
    def classify(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[str, float]:
        """Classify image using ensemble.
        
        Args:
            image: Image to classify
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Collect votes
        votes = {}
        
        for embedder in self.embedders:
            for template in self.prompt_templates:
                predicted_class, confidence = embedder.classify_image(
                    image, self.class_labels, template)
                
                if predicted_class is not None:
                    votes[predicted_class] = votes.get(predicted_class, 0) + confidence
        
        if not votes:
            return None, 0.0
        
        # Find best class
        best_class = max(votes, key=votes.get)
        confidence = votes[best_class] / (len(self.embedders) * len(self.prompt_templates))
        
        return best_class, confidence


def main():
    """Example usage."""
    import argparse
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="CLIP zero-shot classification")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--classes", required=True, help="Comma-separated list of classes")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    
    args = parser.parse_args()
    
    # Load image
    try:
        image = Image.open(args.image).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return
    
    # Parse classes
    class_labels = [c.strip() for c in args.classes.split(',')]
    logger.info(f"Classes: {class_labels}")
    
    # Create CLIP embedder
    embedder = CLIPEmbedder(model_name=args.model)
    
    # Classify image
    predicted_class, confidence = embedder.classify_image(image, class_labels)
    
    logger.info(f"Prediction: {predicted_class} ({confidence:.4f})")
    
    # Get all similarities
    similarities = {}
    prompts = [f"a photo of a {label}" for label in class_labels]
    text_embeddings = embedder.embed_text(prompts)
    image_embedding = embedder.embed_image(image)
    all_similarities = embedder.compute_similarity(image_embedding, text_embeddings)
    
    for i, label in enumerate(class_labels):
        similarities[label] = all_similarities[i]
    
    # Plot results
    plt.figure(figsize=(10, 5))
    
    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.4f}")
    plt.axis('off')
    
    # Show similarities
    plt.subplot(1, 2, 2)
    labels = list(similarities.keys())
    values = [similarities[label] for label in labels]
    
    # Sort by similarity
    sorted_indices = np.argsort(values)[::-1]
    labels = [labels[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Similarity')
    plt.title('CLIP Similarity Scores')
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.splitext(args.image)[0] + '_clip_results.png'
    plt.savefig(output_path)
    logger.info(f"Results saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()