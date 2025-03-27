#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Label management module for semantic understanding.

This handles the mapping between semantic class IDs and labels across
different datasets, along with their attributes and relationships.

Author: Sarah Lin
Date: 2024-01-15
Last modified: 2024-03-02
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Set, Optional, Union, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class LabelManager:
    """Manages semantic labels and their properties across different datasets."""
    
    # Standard datasets and their label files
    DATASET_CONFIGS = {
        'coco': 'coco_labels.json',
        'ade20k': 'ade20k_labels.json',
        'scannet': 'scannet_labels.json',
        'custom': 'custom_labels.json',
    }
    
    def __init__(self, datasets: List[str] = ['coco', 'ade20k'], custom_labels_path: Optional[str] = None):
        """Initialize label manager.
        
        Args:
            datasets: List of datasets to load
            custom_labels_path: Path to custom labels file
        """
        self.labels = {}  # class_id -> label info
        self.label_to_id = {}  # name -> class_id
        self.synonyms = defaultdict(list)  # canonical name -> list of synonyms
        self.category_to_labels = defaultdict(list)  # category -> list of labels
        
        # Load dataset labels
        for dataset in datasets:
            if dataset in self.DATASET_CONFIGS:
                self._load_dataset_labels(dataset)
            else:
                logger.warning(f"Unknown dataset: {dataset}")
        
        # Load custom labels if provided
        if custom_labels_path and os.path.exists(custom_labels_path):
            self._load_custom_labels(custom_labels_path)
        
        logger.info(f"Loaded {len(self.labels)} labels from {len(datasets)} datasets")
        
        # Generate synonyms and categories
        self._generate_synonyms()
        self._categorize_labels()
    
    def _load_dataset_labels(self, dataset: str):
        """Load labels from dataset configuration.
        
        Args:
            dataset: Dataset name
        """
        # Determine config file path
        config_file = self.DATASET_CONFIGS[dataset]
        config_path = os.path.join(os.path.dirname(__file__), 'config', config_file)
        
        # Try to load config file
        try:
            with open(config_path, 'r') as f:
                label_data = json.load(f)
            
            # Process labels
            for item in label_data:
                class_id = item['id']
                label_name = item['name']
                
                # Skip if already loaded with higher priority
                if label_name in self.label_to_id:
                    continue
                
                self.labels[class_id] = {
                    'name': label_name,
                    'dataset': dataset,
                    'color': item.get('color', self._generate_color(class_id)),
                    'category': item.get('category', 'unknown'),
                    'synonyms': item.get('synonyms', []),
                    'attributes': item.get('attributes', {}),
                }
                
                self.label_to_id[label_name] = class_id
            
            logger.info(f"Loaded {len(label_data)} labels from {dataset}")
            
        except Exception as e:
            logger.error(f"Error loading {dataset} labels: {e}")
    
    def _load_custom_labels(self, filepath: str):
        """Load custom labels from file.
        
        Args:
            filepath: Path to custom labels file
        """
        try:
            with open(filepath, 'r') as f:
                label_data = json.load(f)
            
            # Process labels
            for item in label_data:
                class_id = item['id']
                label_name = item['name']
                
                self.labels[class_id] = {
                    'name': label_name,
                    'dataset': 'custom',
                    'color': item.get('color', self._generate_color(class_id)),
                    'category': item.get('category', 'unknown'),
                    'synonyms': item.get('synonyms', []),
                    'attributes': item.get('attributes', {}),
                }
                
                self.label_to_id[label_name] = class_id
            
            logger.info(f"Loaded {len(label_data)} custom labels")
            
        except Exception as e:
            logger.error(f"Error loading custom labels: {e}")
    
    def _generate_color(self, class_id: int) -> List[int]:
        """Generate deterministic color for class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            RGB color as list of 3 integers
        """
        # Use a deterministic hash function based on class_id
        np.random.seed(class_id)
        color = list(np.random.randint(0, 255, 3))
        np.random.seed(None)  # Reset seed
        return color
    
    def _generate_synonyms(self):
        """Generate synonyms for all labels."""
        for class_id, info in self.labels.items():
            label_name = info['name']
            synonyms = info.get('synonyms', [])
            
            # Add label itself to synonyms for lookup
            self.synonyms[label_name].append(label_name)
            
            # Add additional synonyms
            for syn in synonyms:
                if syn not in self.synonyms[label_name]:
                    self.synonyms[label_name].append(syn)
    
    def _categorize_labels(self):
        """Categorize labels by their categories."""
        for class_id, info in self.labels.items():
            category = info.get('category', 'unknown')
            label_name = info['name']
            
            self.category_to_labels[category].append(label_name)
    
    def get_label_info(self, class_id: int) -> Optional[Dict]:
        """Get information for a class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            Label information or None if not found
        """
        return self.labels.get(class_id)
    
    def get_class_id(self, label: str) -> Optional[int]:
        """Get class ID for a label name.
        
        Args:
            label: Label name
            
        Returns:
            Class ID or None if not found
        """
        # Direct lookup
        if label in self.label_to_id:
            return self.label_to_id[label]
        
        # Search in synonyms
        label_lower = label.lower()
        for canonical, synonyms in self.synonyms.items():
            if label_lower in [s.lower() for s in synonyms]:
                return self.label_to_id.get(canonical)
        
        # Try partial match
        for canonical in self.label_to_id:
            if label_lower in canonical.lower() or canonical.lower() in label_lower:
                return self.label_to_id[canonical]
        
        return None
    
    def get_color(self, class_id: int) -> List[int]:
        """Get color for a class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            RGB color as list of 3 integers
        """
        info = self.labels.get(class_id)
        if info and 'color' in info:
            return info['color']
        else:
            return self._generate_color(class_id)
    
    def get_category(self, class_id: int) -> str:
        """Get category for a class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            Category name
        """
        info = self.labels.get(class_id)
        if info and 'category' in info:
            return info['category']
        else:
            return 'unknown'
    
    def get_labels_in_category(self, category: str) -> List[str]:
        """Get all labels in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of label names
        """
        return self.category_to_labels.get(category, [])
    
    def get_potential_classes(self, label: str) -> List[str]:
        """Get potential class names for a given label.
        
        This is useful for zero-shot recognition, when we want to
        suggest alternative class labels for a detected object.
        
        Args:
            label: Current label
            
        Returns:
            List of potential class names
        """
        # Start with the label itself
        if label not in self.label_to_id:
            return []
            
        class_id = self.label_to_id[label]
        category = self.get_category(class_id)
        
        # Get all labels in the same category
        potentials = self.get_labels_in_category(category)
        
        # Add synonyms
        synonyms = self.synonyms.get(label, [])
        for syn in synonyms:
            if syn not in potentials:
                potentials.append(syn)
        
        # Ensure the original label is included
        if label not in potentials:
            potentials.append(label)
        
        return potentials
    
    def create_label_mapping(self) -> Dict[int, Dict]:
        """Create mapping of all labels with their properties.
        
        Returns:
            Dictionary mapping class IDs to label properties
        """
        return self.labels


# Default COCO labels if config files not available
DEFAULT_COCO_LABELS = [
    {"id": 1, "name": "person", "category": "living"},
    {"id": 2, "name": "bicycle", "category": "vehicle"},
    {"id": 3, "name": "car", "category": "vehicle"},
    {"id": 4, "name": "motorcycle", "category": "vehicle"},
    {"id": 5, "name": "airplane", "category": "vehicle"},
    {"id": 6, "name": "bus", "category": "vehicle"},
    {"id": 7, "name": "train", "category": "vehicle"},
    {"id": 8, "name": "truck", "category": "vehicle"},
    {"id": 9, "name": "boat", "category": "vehicle"},
    {"id": 10, "name": "traffic light", "category": "outdoor"},
    {"id": 11, "name": "fire hydrant", "category": "outdoor"},
    {"id": 13, "name": "stop sign", "category": "outdoor"},
    {"id": 14, "name": "parking meter", "category": "outdoor"},
    {"id": 15, "name": "bench", "category": "furniture"},
    {"id": 16, "name": "bird", "category": "animal"},
    {"id": 17, "name": "cat", "category": "animal"},
    {"id": 18, "name": "dog", "category": "animal"},
    {"id": 19, "name": "horse", "category": "animal"},
    {"id": 20, "name": "sheep", "category": "animal"},
    {"id": 21, "name": "cow", "category": "animal"},
    {"id": 22, "name": "elephant", "category": "animal"},
    {"id": 23, "name": "bear", "category": "animal"},
    {"id": 24, "name": "zebra", "category": "animal"},
    {"id": 25, "name": "giraffe", "category": "animal"},
    {"id": 27, "name": "backpack", "category": "accessory"},
    {"id": 28, "name": "umbrella", "category": "accessory"},
    {"id": 31, "name": "handbag", "category": "accessory"},
    {"id": 32, "name": "tie", "category": "accessory"},
    {"id": 33, "name": "suitcase", "category": "accessory"},
    {"id": 34, "name": "frisbee", "category": "sports"},
    {"id": 35, "name": "skis", "category": "sports"},
    {"id": 36, "name": "snowboard", "category": "sports"},
    {"id": 37, "name": "sports ball", "category": "sports"},
    {"id": 38, "name": "kite", "category": "sports"},
    {"id": 39, "name": "baseball bat", "category": "sports"},
    {"id": 40, "name": "baseball glove", "category": "sports"},
    {"id": 41, "name": "skateboard", "category": "sports"},
    {"id": 42, "name": "surfboard", "category": "sports"},
    {"id": 43, "name": "tennis racket", "category": "sports"},
    {"id": 44, "name": "bottle", "category": "kitchen"},
    {"id": 46, "name": "wine glass", "category": "kitchen"},
    {"id": 47, "name": "cup", "category": "kitchen"},
    {"id": 48, "name": "fork", "category": "kitchen"},
    {"id": 49, "name": "knife", "category": "kitchen"},
    {"id": 50, "name": "spoon", "category": "kitchen"},
    {"id": 51, "name": "bowl", "category": "kitchen"},
    {"id": 52, "name": "banana", "category": "food"},
    {"id": 53, "name": "apple", "category": "food"},
    {"id": 54, "name": "sandwich", "category": "food"},
    {"id": 55, "name": "orange", "category": "food"},
    {"id": 56, "name": "broccoli", "category": "food"},
    {"id": 57, "name": "carrot", "category": "food"},
    {"id": 58, "name": "hot dog", "category": "food"},
    {"id": 59, "name": "pizza", "category": "food"},
    {"id": 60, "name": "donut", "category": "food"},
    {"id": 61, "name": "cake", "category": "food"},
    {"id": 62, "name": "chair", "category": "furniture"},
    {"id": 63, "name": "couch", "category": "furniture"},
    {"id": 64, "name": "potted plant", "category": "furniture"},
    {"id": 65, "name": "bed", "category": "furniture"},
    {"id": 67, "name": "dining table", "category": "furniture"},
    {"id": 70, "name": "toilet", "category": "furniture"},
    {"id": 72, "name": "tv", "category": "electronics"},
    {"id": 73, "name": "laptop", "category": "electronics"},
    {"id": 74, "name": "mouse", "category": "electronics"},
    {"id": 75, "name": "remote", "category": "electronics"},
    {"id": 76, "name": "keyboard", "category": "electronics"},
    {"id": 77, "name": "cell phone", "category": "electronics"},
    {"id": 78, "name": "microwave", "category": "appliance"},
    {"id": 79, "name": "oven", "category": "appliance"},
    {"id": 80, "name": "toaster", "category": "appliance"},
    {"id": 81, "name": "sink", "category": "appliance"},
    {"id": 82, "name": "refrigerator", "category": "appliance"},
    {"id": 84, "name": "book", "category": "indoor"},
    {"id": 85, "name": "clock", "category": "indoor"},
    {"id": 86, "name": "vase", "category": "indoor"},
    {"id": 87, "name": "scissors", "category": "indoor"},
    {"id": 88, "name": "teddy bear", "category": "indoor"},
    {"id": 89, "name": "hair drier", "category": "indoor"},
    {"id": 90, "name": "toothbrush", "category": "indoor"},
]

# Function to ensure label config files exist
def create_default_label_configs():
    """Create default label configuration files if they don't exist."""
    # Create config directory
    config_dir = os.path.join(os.path.dirname(__file__), 'config')
    os.makedirs(config_dir, exist_ok=True)
    
    # Create COCO labels file
    coco_path = os.path.join(config_dir, 'coco_labels.json')
    if not os.path.exists(coco_path):
        try:
            with open(coco_path, 'w') as f:
                json.dump(DEFAULT_COCO_LABELS, f, indent=2)
            logger.info(f"Created default COCO labels file at {coco_path}")
        except Exception as e:
            logger.error(f"Failed to create default COCO labels file: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Ensure config files exist
    create_default_label_configs()
    
    # Test label manager
    label_manager = LabelManager()
    
    # Print some stats
    print(f"Loaded {len(label_manager.labels)} labels")
    print(f"Categories: {list(label_manager.category_to_labels.keys())}")
    
    # Test lookups
    test_labels = ['chair', 'sofa', 'person', 'car']
    for label in test_labels:
        class_id = label_manager.get_class_id(label)
        if class_id is not None:
            info = label_manager.get_label_info(class_id)
            category = label_manager.get_category(class_id)
            color = label_manager.get_color(class_id)
            
            print(f"{label} (ID: {class_id})")
            print(f"  Category: {category}")
            print(f"  Color: {color}")
            
            potentials = label_manager.get_potential_classes(label)
            print(f"  Potential classes: {potentials[:5]}...")
        else:
            print(f"{label}: Not found")