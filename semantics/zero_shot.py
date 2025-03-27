#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Zero-shot recognition module using CLIP embeddings.

This module provides functionality for recognizing objects not seen during training,
by leveraging CLIP's vision-language multimodal understanding.

Author: Sarah Lin
Date: 2024-01-25
Last modified: 2024-03-15
"""

import os
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
from dataclasses import dataclass

from recontext.semantics.clip_embeddings import CLIPEmbedder
from recontext.semantics.label_manager import LabelManager

logger = logging.getLogger(__name__)

@dataclass
class ZeroShotPrediction:
    """Container for zero-shot prediction results."""
    label: str  # Predicted class name
    score: float  # Confidence score
    all_scores: Dict[str, float]  # Scores for all candidate classes
    embedding: Optional[np.ndarray] = None  # CLIP embedding if available


class ZeroShotRecognizer:
    """Zero-shot recognition using CLIP."""
    
    def __init__(self, 
                candidate_labels: Optional[List[str]] = None,
                clip_model: str = "ViT-B/32",
                device: Optional[str] = None,
                threshold: float = 0.2):
        """Initialize zero-shot recognizer.
        
        Args:
            candidate_labels: List of candidate class labels (if None, uses label manager)
            clip_model: CLIP model name
            device: Device to use (cuda or cpu)
            threshold: Confidence threshold for predictions
        """
        self.threshold = threshold
        
        # Initialize CLIP embedder
        try:
            self.embedder = CLIPEmbedder(model_name=clip_model, device=device)
            logger.info(f"Initialized CLIP model: {clip_model}")
        except ImportError:
            raise ImportError("CLIP is required for zero-shot recognition")
        
        # Initialize label manager
        self.label_manager = LabelManager()
        
        # Set candidate labels
        if candidate_labels is not None:
            self.candidate_labels = candidate_labels
        else:
            # Get all labels from label manager
            all_labels = list(self.label_manager.label_to_id.keys())
            # Filter out duplicates and ensure we have a reasonable set
            self.candidate_labels = list(set(all_labels))
        
        logger.info(f"Using {len(self.candidate_labels)} candidate labels for zero-shot recognition")
        
        # Precompute text embeddings for candidate labels
        self._precompute_text_embeddings()
    
    def _precompute_text_embeddings(self):
        """Precompute text embeddings for all candidate labels."""
        # Create prompt templates
        prompt_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "a {} in the scene",
            "an image of a {}"
        ]
        
        # Special handling for certain categories
        for label in self.candidate_labels:
            if label in ["sky", "ground", "water", "road", "grass", "sand", "snow", "floor", "wall", "ceiling"]:
                prompt_templates.append("a photo of {}")
                prompt_templates.append("a picture of {}")
                prompt_templates.append("an image showing {}")
        
        # Generate prompts and compute embeddings
        all_prompts = []
        prompt_to_label = {}
        
        for label in self.candidate_labels:
            for template in prompt_templates:
                prompt = template.format(label)
                all_prompts.append(prompt)
                prompt_to_label[prompt] = label
        
        # Compute embeddings in batches to avoid memory issues
        batch_size = 128
        all_embeddings = []
        
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            batch_embeddings = self.embedder.embed_text(batch_prompts)
            
            # Stack if there are multiple embeddings
            if len(batch_embeddings.shape) == 1:
                batch_embeddings = batch_embeddings.reshape(1, -1)
                
            all_embeddings.append(batch_embeddings)
        
        # Combine embeddings
        self.text_embeddings = np.concatenate(all_embeddings, axis=0)
        self.prompts = all_prompts
        self.prompt_to_label = prompt_to_label
        
        logger.info(f"Precomputed {len(all_prompts)} text embeddings")
    
    def predict(self, image: Union[str, np.ndarray, Image.Image],
               candidate_labels: Optional[List[str]] = None) -> ZeroShotPrediction:
        """Predict class for image using zero-shot recognition.
        
        Args:
            image: Input image (path, numpy array, or PIL image)
            candidate_labels: Optional subset of candidate labels to consider
            
        Returns:
            Zero-shot prediction
        """
        # Get image embedding
        embedding = self.embedder.embed_image(image)
        
        # Handle the case where candidate_labels is provided
        if candidate_labels is not None and candidate_labels != self.candidate_labels:
            # Generate embeddings for these specific candidates
            all_prompts = []
            prompt_to_label = {}
            
            for label in candidate_labels:
                prompt = f"a photo of a {label}"
                all_prompts.append(prompt)
                prompt_to_label[prompt] = label
            
            text_embeddings = self.embedder.embed_text(all_prompts)
            
            # Reshape if needed
            if len(text_embeddings.shape) == 1:
                text_embeddings = text_embeddings.reshape(1, -1)
            
            # Compute similarities
            similarities = self.embedder.compute_similarity(embedding, text_embeddings)
            
            # Get predictions
            all_scores = {}
            for i, prompt in enumerate(all_prompts):
                label = prompt_to_label[prompt]
                score = float(similarities[i])
                
                if label in all_scores:
                    all_scores[label] = max(all_scores[label], score)
                else:
                    all_scores[label] = score
        else:
            # Use precomputed embeddings
            
            # Compute similarities
            similarities = self.embedder.compute_similarity(embedding, self.text_embeddings)
            
            # Group by label and take max score
            all_scores = {}
            for i, prompt in enumerate(self.prompts):
                label = self.prompt_to_label[prompt]
                score = float(similarities[i])
                
                if label in all_scores:
                    all_scores[label] = max(all_scores[label], score)
                else:
                    all_scores[label] = score
        
        # Sort by score (descending)
        sorted_labels = sorted(all_scores.keys(), key=lambda k: all_scores[k], reverse=True)
        
        # Get top prediction
        if sorted_labels:
            top_label = sorted_labels[0]
            top_score = all_scores[top_label]
            
            # Apply threshold
            if top_score < self.threshold:
                top_label = "unknown"
                top_score = 0.0
        else:
            top_label = "unknown"
            top_score = 0.0
        
        return ZeroShotPrediction(
            label=top_label,
            score=top_score,
            all_scores=all_scores,
            embedding=embedding
        )
    
    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]],
                     candidate_labels: Optional[List[str]] = None) -> List[ZeroShotPrediction]:
        """Predict classes for multiple images.
        
        Args:
            images: List of input images
            candidate_labels: Optional subset of candidate labels to consider
            
        Returns:
            List of zero-shot predictions
        """
        predictions = []
        
        for image in images:
            prediction = self.predict(image, candidate_labels)
            predictions.append(prediction)
        
        return predictions
    
    def get_similar_labels(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get similar labels for an embedding.
        
        Args:
            embedding: Image embedding
            top_k: Number of top labels to return
            
        Returns:
            List of (label, score) tuples
        """
        # Compute similarities with all text embeddings
        similarities = self.embedder.compute_similarity(embedding, self.text_embeddings)
        
        # Group by label and take max score
        all_scores = {}
        for i, prompt in enumerate(self.prompts):
            label = self.prompt_to_label[prompt]
            score = float(similarities[i])
            
            if label in all_scores:
                all_scores[label] = max(all_scores[label], score)
            else:
                all_scores[label] = score
        
        # Sort by score (descending)
        sorted_labels = sorted(all_scores.keys(), key=lambda k: all_scores[k], reverse=True)
        
        # Return top-k
        return [(label, all_scores[label]) for label in sorted_labels[:top_k]]


class HierarchicalZeroShotRecognizer:
    """Hierarchical zero-shot recognition using categories."""
    
    def __init__(self, 
                clip_model: str = "ViT-B/32",
                device: Optional[str] = None,
                threshold: float = 0.2):
        """Initialize hierarchical zero-shot recognizer.
        
        Args:
            clip_model: CLIP model name
            device: Device to use (cuda or cpu)
            threshold: Confidence threshold for predictions
        """
        self.threshold = threshold
        
        # Initialize label manager
        self.label_manager = LabelManager()
        
        # Get categories from label manager
        self.categories = list(self.label_manager.category_to_labels.keys())
        logger.info(f"Found {len(self.categories)} categories")
        
        # Initialize category recognizer
        self.category_recognizer = ZeroShotRecognizer(
            candidate_labels=self.categories,
            clip_model=clip_model,
            device=device,
            threshold=threshold
        )
        
        # Initialize recognizers for each category
        self.category_recognizers = {}
        for category in self.categories:
            labels = self.label_manager.get_labels_in_category(category)
            if labels:
                self.category_recognizers[category] = ZeroShotRecognizer(
                    candidate_labels=labels,
                    clip_model=clip_model,
                    device=device,
                    threshold=threshold
                )
                logger.info(f"Initialized recognizer for category {category} with {len(labels)} labels")
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> ZeroShotPrediction:
        """Predict class for image using hierarchical zero-shot recognition.
        
        Args:
            image: Input image (path, numpy array, or PIL image)
            
        Returns:
            Zero-shot prediction
        """
        # First predict category
        category_prediction = self.category_recognizer.predict(image)
        
        # If category confidence is too low, return unknown
        if category_prediction.score < self.threshold:
            return ZeroShotPrediction(
                label="unknown",
                score=0.0,
                all_scores=category_prediction.all_scores,
                embedding=category_prediction.embedding
            )
        
        # Get predicted category
        category = category_prediction.label
        
        # If we have a recognizer for this category, use it
        if category in self.category_recognizers:
            # Use the embedding from the category prediction
            class_prediction = self.category_recognizers[category].predict(image)
            
            # Adjust score by category confidence
            class_prediction.score *= category_prediction.score
            
            return class_prediction
        else:
            # If no recognizer for this category, return the category itself
            return category_prediction


def main():
    """Example usage for demonstration."""
    import argparse
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Zero-shot recognition")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default="ViT-B/32", help="CLIP model name")
    parser.add_argument("--hierarchical", action="store_true", help="Use hierarchical recognition")
    parser.add_argument("--labels", help="Comma-separated list of candidate labels")
    
    args = parser.parse_args()
    
    # Load image
    image = Image.open(args.image)
    
    # Parse candidate labels if provided
    candidate_labels = None
    if args.labels:
        candidate_labels = [label.strip() for label in args.labels.split(",")]
    
    # Initialize recognizer
    if args.hierarchical:
        recognizer = HierarchicalZeroShotRecognizer(clip_model=args.model)
    else:
        recognizer = ZeroShotRecognizer(candidate_labels=candidate_labels, clip_model=args.model)
    
    # Run prediction
    prediction = recognizer.predict(image)
    
    # Print results
    print(f"Prediction: {prediction.label} (score: {prediction.score:.3f})")
    
    # Print top-5 predictions
    print("\nTop 5 predictions:")
    for label, score in sorted(prediction.all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {label}: {score:.3f}")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {prediction.label}\nScore: {prediction.score:.3f}")
    plt.axis('off')
    
    # Show top-5 scores
    plt.subplot(1, 2, 2)
    top_labels = sorted(prediction.all_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    labels = [item[0] for item in top_labels]
    scores = [item[1] for item in top_labels]
    
    plt.barh(range(len(labels)), scores)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Score')
    plt.title('Top 5 predictions')
    
    plt.tight_layout()
    
    # Save or show
    plt.savefig("zero_shot_result.png")
    print("Results saved to zero_shot_result.png")
    plt.show()


if __name__ == "__main__":
    main()