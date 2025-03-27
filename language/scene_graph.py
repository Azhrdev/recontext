#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scene Graph generation module for 3D semantic understanding.

This module constructs a scene graph representing objects and their spatial/functional
relationships within the 3D scene, enabling semantic reasoning and natural language queries.

Author: Michael Zhang
Date: 2024-02-20
"""

import numpy as np
import open3d as o3d
import networkx as nx
import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from recontext.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

@dataclass
class Object3D:
    """Representation of a 3D object in the scene."""
    id: int
    label: str
    class_id: int
    points: np.ndarray  # Nx3 array of points
    center: np.ndarray  # 3D center point
    bbox: np.ndarray  # 3D axis-aligned bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
    color: Tuple[int, int, int]
    point_indices: List[int] = field(default_factory=list)  # Indices in the original point cloud
    attributes: Dict[str, any] = field(default_factory=dict)  # Additional attributes (material, etc.)
    
    @property
    def size(self) -> np.ndarray:
        """Get object dimensions (width, height, depth)."""
        return self.bbox[3:6] - self.bbox[0:3]
    
    @property
    def volume(self) -> float:
        """Get approximate object volume."""
        return np.prod(self.size)
    
    @property
    def point_count(self) -> int:
        """Get number of points in the object."""
        return len(self.points)


@dataclass
class Relationship:
    """Representation of a relationship between objects."""
    id: int
    type: str  # Relationship type (spatial, functional, etc.)
    source_id: int  # ID of source object
    target_id: int  # ID of target object
    attributes: Dict[str, any] = field(default_factory=dict)  # Additional attributes
    confidence: float = 1.0  # Confidence score for this relationship


class SceneGraph:
    """Scene graph representation of 3D semantic understanding."""
    
    # Constants for spatial relationships
    SPATIAL_RELATIONSHIPS = [
        'above', 'below', 'left_of', 'right_of', 'in_front_of', 'behind',
        'inside', 'contains', 'on_top_of', 'under', 'next_to', 'near'
    ]
    
    # Constants for support relationships
    SUPPORT_RELATIONSHIPS = [
        'supports', 'supported_by'
    ]
    
    def __init__(self):
        """Initialize an empty scene graph."""
        self.objects = {}  # id -> Object3D
        self.relationships = {}  # id -> Relationship
        
        # Build NetworkX graph for easier traversal
        self.graph = nx.DiGraph()
        
        # Dictionary for fast spatial lookups
        self.spatial_index = None
        
        # Global scene information
        self.scene_info = {
            'name': 'Unnamed Scene',
            'dimensions': None,  # [width, height, depth]
            'global_bbox': None,  # [min_x, min_y, min_z, max_x, max_y, max_z]
            'stats': {},  # Various statistics
        }
    
    def add_object(self, obj: Object3D) -> int:
        """Add an object to the scene graph.
        
        Args:
            obj: Object to add
            
        Returns:
            Object ID
        """
        self.objects[obj.id] = obj
        
        # Add to NetworkX graph
        self.graph.add_node(obj.id, 
                          type='object',
                          label=obj.label,
                          class_id=obj.class_id,
                          center=obj.center.tolist(),
                          bbox=obj.bbox.tolist())
        
        # Invalidate spatial index since objects changed
        self.spatial_index = None
        
        return obj.id
    
    def add_relationship(self, rel: Relationship) -> int:
        """Add a relationship to the scene graph.
        
        Args:
            rel: Relationship to add
            
        Returns:
            Relationship ID
        """
        self.relationships[rel.id] = rel
        
        # Add to NetworkX graph
        self.graph.add_edge(rel.source_id, 
                          rel.target_id,
                          id=rel.id,
                          type=rel.type,
                          attributes=rel.attributes,
                          confidence=rel.confidence)
        
        return rel.id
    
    def remove_object(self, obj_id: int) -> bool:
        """Remove an object from the scene graph.
        
        Args:
            obj_id: ID of object to remove
            
        Returns:
            True if object was removed, False if not found
        """
        if obj_id not in self.objects:
            return False
        
        # Remove all relationships involving this object
        rel_ids_to_remove = []
        for rel_id, rel in self.relationships.items():
            if rel.source_id == obj_id or rel.target_id == obj_id:
                rel_ids_to_remove.append(rel_id)
        
        for rel_id in rel_ids_to_remove:
            del self.relationships[rel_id]
        
        # Remove from NetworkX graph
        if self.graph.has_node(obj_id):
            self.graph.remove_node(obj_id)  # This also removes all connected edges
        
        # Remove object
        del self.objects[obj_id]
        
        # Invalidate spatial index
        self.spatial_index = None
        
        return True
    
    def remove_relationship(self, rel_id: int) -> bool:
        """Remove a relationship from the scene graph.
        
        Args:
            rel_id: ID of relationship to remove
            
        Returns:
            True if relationship was removed, False if not found
        """
        if rel_id not in self.relationships:
            return False
        
        # Get source and target
        rel = self.relationships[rel_id]
        
        # Remove from NetworkX graph
        if self.graph.has_edge(rel.source_id, rel.target_id):
            # Check if this is the right edge (there could be multiple edges between same nodes)
            edge_data = self.graph.get_edge_data(rel.source_id, rel.target_id)
            if edge_data and edge_data.get('id') == rel_id:
                self.graph.remove_edge(rel.source_id, rel.target_id)
        
        # Remove relationship
        del self.relationships[rel_id]
        
        return True
    
    def get_object(self, obj_id: int) -> Optional[Object3D]:
        """Get object by ID.
        
        Args:
            obj_id: Object ID
            
        Returns:
            Object if found, None otherwise
        """
        return self.objects.get(obj_id)
    
    def get_relationship(self, rel_id: int) -> Optional[Relationship]:
        """Get relationship by ID.
        
        Args:
            rel_id: Relationship ID
            
        Returns:
            Relationship if found, None otherwise
        """
        return self.relationships.get(rel_id)
    
    def get_objects_by_label(self, label: str) -> List[Object3D]:
        """Get all objects with the given label.
        
        Args:
            label: Object label
            
        Returns:
            List of matching objects
        """
        return [obj for obj in self.objects.values() if obj.label == label]
    
    def get_relationships_between(self, source_id: int, target_id: int) -> List[Relationship]:
        """Get all relationships between two objects.
        
        Args:
            source_id: Source object ID
            target_id: Target object ID
            
        Returns:
            List of relationships
        """
        return [rel for rel in self.relationships.values() 
                if rel.source_id == source_id and rel.target_id == target_id]
    
    def get_neighbors(self, obj_id: int, rel_type: Optional[str] = None) -> List[int]:
        """Get neighboring object IDs.
        
        Args:
            obj_id: Object ID
            rel_type: Optional relationship type filter
            
        Returns:
            List of neighbor object IDs
        """
        if not self.graph.has_node(obj_id):
            return []
        
        if rel_type is None:
            # All neighbors
            return list(self.graph.successors(obj_id))
        else:
            # Filter by relationship type
            neighbors = []
            for _, neighbor, data in self.graph.out_edges(obj_id, data=True):
                if data.get('type') == rel_type:
                    neighbors.append(neighbor)
            return neighbors
    
    def build_spatial_index(self):
        """Build spatial index for efficient spatial queries."""
        if not self.objects:
            return
        
        try:
            import rtree
            
            # Create R-tree index
            p = rtree.index.Property()
            p.dimension = 3
            self.spatial_index = rtree.index.Index(properties=p)
            
            # Add objects to index
            for obj_id, obj in self.objects.items():
                # Use bounding box for indexing
                self.spatial_index.insert(obj_id, 
                                       (obj.bbox[0], obj.bbox[1], obj.bbox[2],
                                        obj.bbox[3], obj.bbox[4], obj.bbox[5]))
                
            logger.info(f"Built spatial index with {len(self.objects)} objects")
            
        except ImportError:
            logger.warning("rtree package not found. Spatial queries will be slower.")
            self.spatial_index = None
    
    def get_objects_in_region(self, bbox: np.ndarray) -> List[int]:
        """Get objects within or intersecting a 3D region.
        
        Args:
            bbox: Bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
            
        Returns:
            List of object IDs
        """
        if self.spatial_index is None:
            # Rebuild index if needed
            self.build_spatial_index()
            
            # If still None (import failed), use brute force
            if self.spatial_index is None:
                return self._brute_force_region_query(bbox)
        
        # Query spatial index
        return list(self.spatial_index.intersection(
            (bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5])
        ))
    
    def _brute_force_region_query(self, bbox: np.ndarray) -> List[int]:
        """Fallback method for region queries without spatial index.
        
        Args:
            bbox: Bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
            
        Returns:
            List of object IDs
        """
        results = []
        
        for obj_id, obj in self.objects.items():
            # Check for overlap between bounding boxes
            if (obj.bbox[0] <= bbox[3] and obj.bbox[3] >= bbox[0] and
                obj.bbox[1] <= bbox[4] and obj.bbox[4] >= bbox[1] and
                obj.bbox[2] <= bbox[5] and obj.bbox[5] >= bbox[2]):
                results.append(obj_id)
                
        return results
    
    def infer_spatial_relationships(self, 
                                   distance_threshold: float = 0.5,
                                   min_confidence: float = 0.7) -> List[Relationship]:
        """Infer spatial relationships between objects.
        
        Args:
            distance_threshold: Maximum distance for 'near' relationship
            min_confidence: Minimum confidence for relationships
            
        Returns:
            List of inferred relationships
        """
        logger.info("Inferring spatial relationships")
        
        # Get all object pairs
        object_ids = list(self.objects.keys())
        relationships = []
        
        # Calculate global Y-up direction (assuming Y is up)
        up_vector = np.array([0, 1, 0])
        
        # Generate a unique ID for new relationships
        next_rel_id = max(self.relationships.keys(), default=0) + 1
        
        # Check each object pair
        for i, obj1_id in enumerate(tqdm(object_ids, desc="Processing objects")):
            obj1 = self.objects[obj1_id]
            
            for j, obj2_id in enumerate(object_ids):
                if i == j:
                    continue  # Skip self
                    
                obj2 = self.objects[obj2_id]
                
                # Vector from obj1 to obj2
                direction = obj2.center - obj1.center
                distance = np.linalg.norm(direction)
                
                # Skip if too far apart
                if distance > distance_threshold * max(np.max(obj1.size), np.max(obj2.size)):
                    continue
                
                # Normalize direction
                if distance > 0:
                    direction = direction / distance
                
                # Check for various spatial relationships
                relationships_to_add = []
                
                # 1. Above/below (Y axis)
                y_diff = obj2.center[1] - obj1.center[1]
                if y_diff > 0.5 * (obj1.size[1] + obj2.size[1]):
                    # obj2 is above obj1
                    relationships_to_add.append(('above', obj2_id, obj1_id))
                elif y_diff < -0.5 * (obj1.size[1] + obj2.size[1]):
                    # obj2 is below obj1
                    relationships_to_add.append(('below', obj2_id, obj1_id))
                
                # 2. On top of / under (more specific than above/below)
                if y_diff > 0 and self._check_vertical_alignment(obj1, obj2):
                    # Check if obj2 is on top of obj1
                    if abs(obj1.bbox[4] - obj2.bbox[1]) < 0.1:  # Y-max of obj1 close to Y-min of obj2
                        relationships_to_add.append(('on_top_of', obj2_id, obj1_id))
                        relationships_to_add.append(('supports', obj1_id, obj2_id))
                elif y_diff < 0 and self._check_vertical_alignment(obj1, obj2):
                    # Check if obj2 is under obj1
                    if abs(obj1.bbox[1] - obj2.bbox[4]) < 0.1:  # Y-min of obj1 close to Y-max of obj2
                        relationships_to_add.append(('under', obj2_id, obj1_id))
                        relationships_to_add.append(('supports', obj2_id, obj1_id))
                
                # 3. Left/right (X axis)
                x_diff = obj2.center[0] - obj1.center[0]
                if abs(x_diff) > 0.5 * (obj1.size[0] + obj2.size[0]):
                    if x_diff > 0:
                        relationships_to_add.append(('right_of', obj2_id, obj1_id))
                        relationships_to_add.append(('left_of', obj1_id, obj2_id))
                
                # 4. In front of / behind (Z axis)
                z_diff = obj2.center[2] - obj1.center[2]
                if abs(z_diff) > 0.5 * (obj1.size[2] + obj2.size[2]):
                    if z_diff > 0:
                        relationships_to_add.append(('behind', obj2_id, obj1_id))
                        relationships_to_add.append(('in_front_of', obj1_id, obj2_id))
                
                # 5. Inside / contains
                if self._check_containment(obj1, obj2):
                    relationships_to_add.append(('contains', obj1_id, obj2_id))
                    relationships_to_add.append(('inside', obj2_id, obj1_id))
                
                # 6. Near
                if distance < distance_threshold:
                    relationships_to_add.append(('near', obj1_id, obj2_id))
                    relationships_to_add.append(('near', obj2_id, obj1_id))
                
                # Add relationships with confidence scores
                for rel_type, source_id, target_id in relationships_to_add:
                    # Calculate confidence based on criteria
                    confidence = self._calculate_relationship_confidence(
                        rel_type, self.objects[source_id], self.objects[target_id])
                    
                    if confidence >= min_confidence:
                        rel = Relationship(
                            id=next_rel_id,
                            type=rel_type,
                            source_id=source_id,
                            target_id=target_id,
                            confidence=confidence
                        )
                        relationships.append(rel)
                        next_rel_id += 1
        
        # Add relationships to the graph
        for rel in relationships:
            self.add_relationship(rel)
            
        logger.info(f"Inferred {len(relationships)} spatial relationships")
        
        return relationships
    
    def _check_vertical_alignment(self, obj1: Object3D, obj2: Object3D) -> bool:
        """Check if two objects are vertically aligned (for on_top_of relationship).
        
        Args:
            obj1: First object
            obj2: Second object
            
        Returns:
            True if objects are vertically aligned
        """
        # Check if there's overlap in XZ plane
        x_overlap = min(obj1.bbox[3], obj2.bbox[3]) - max(obj1.bbox[0], obj2.bbox[0])
        z_overlap = min(obj1.bbox[5], obj2.bbox[5]) - max(obj1.bbox[2], obj2.bbox[2])
        
        return x_overlap > 0 and z_overlap > 0
    
    def _check_containment(self, obj1: Object3D, obj2: Object3D) -> bool:
        """Check if obj1 contains obj2.
        
        Args:
            obj1: Container object
            obj2: Contained object
            
        Returns:
            True if obj1 contains obj2
        """
        # Check if obj2's bounding box is inside obj1's bounding box
        return (obj1.bbox[0] < obj2.bbox[0] and obj1.bbox[3] > obj2.bbox[3] and
                obj1.bbox[1] < obj2.bbox[1] and obj1.bbox[4] > obj2.bbox[4] and
                obj1.bbox[2] < obj2.bbox[2] and obj1.bbox[5] > obj2.bbox[5])
    
    def _calculate_relationship_confidence(self, rel_type: str, obj1: Object3D, obj2: Object3D) -> float:
        """Calculate confidence score for a spatial relationship.
        
        Args:
            rel_type: Relationship type
            obj1: Source object
            obj2: Target object
            
        Returns:
            Confidence score between 0 and 1
        """
        # Different criteria for different relationship types
        if rel_type == 'on_top_of' or rel_type == 'under':
            # Higher confidence when objects are close vertically
            y_distance = abs(obj1.bbox[4] - obj2.bbox[1]) if rel_type == 'on_top_of' else abs(obj1.bbox[1] - obj2.bbox[4])
            return max(0, 1.0 - y_distance)
        
        elif rel_type == 'inside' or rel_type == 'contains':
            # Higher confidence when contained object is well within container
            margin = min(
                obj1.bbox[0] - obj2.bbox[0], obj2.bbox[3] - obj1.bbox[3],
                obj1.bbox[1] - obj2.bbox[1], obj2.bbox[4] - obj1.bbox[4],
                obj1.bbox[2] - obj2.bbox[2], obj2.bbox[5] - obj1.bbox[5]
            )
            return max(0, min(1.0, margin / max(obj2.size)))
        
        elif rel_type == 'near':
            # Higher confidence when objects are closer
            distance = np.linalg.norm(obj1.center - obj2.center)
            max_size = max(np.max(obj1.size), np.max(obj2.size))
            return max(0, 1.0 - distance / max_size)
        
        else:
            # Default confidence for other relationships
            return 0.8
    
    def infer_functional_relationships(self) -> List[Relationship]:
        """Infer functional relationships based on object types and spatial relationships.
        
        Returns:
            List of inferred relationships
        """
        logger.info("Inferring functional relationships")
        
        # Common functional relationships
        functional_rules = [
            # (source_type, target_type, spatial_rel, functional_rel, confidence)
            ('table', 'chair', 'near', 'used_with', 0.9),
            ('desk', 'chair', 'near', 'used_with', 0.9),
            ('shelf', '*', 'contains', 'stores', 0.95),
            ('cabinet', '*', 'contains', 'stores', 0.95),
            ('refrigerator', 'food', 'contains', 'stores', 0.95),
            ('cup', 'liquid', 'contains', 'contains', 0.9),
            ('bowl', 'food', 'contains', 'contains', 0.9),
            ('lamp', '*', 'on_top_of', 'illuminates', 0.8),
            ('tv', 'remote', 'near', 'controlled_by', 0.8),
            ('door', 'room', 'near', 'leads_to', 0.7),
            ('window', 'wall', 'inside', 'part_of', 0.9),
        ]
        
        # Generate a unique ID for new relationships
        next_rel_id = max(self.relationships.keys(), default=0) + 1
        
        # Apply rules to infer functional relationships
        inferred_relationships = []
        
        # First, collect existing spatial relationships
        spatial_rels = {}
        for rel in self.relationships.values():
            if rel.type in self.SPATIAL_RELATIONSHIPS:
                key = (rel.source_id, rel.target_id)
                if key not in spatial_rels or rel.confidence > spatial_rels[key][1]:
                    spatial_rels[key] = (rel.type, rel.confidence)
        
        # Apply rules
        for source_id, obj1 in self.objects.items():
            for target_id, obj2 in self.objects.items():
                if source_id == target_id:
                    continue
                
                # Check if there's a spatial relationship
                key = (source_id, target_id)
                if key not in spatial_rels:
                    continue
                
                spatial_rel, spatial_conf = spatial_rels[key]
                
                # Check rules
                for src_type, tgt_type, req_spatial, func_rel, base_conf in functional_rules:
                    # Match object types
                    src_match = (src_type == '*' or obj1.label.lower() == src_type.lower() or 
                                src_type.lower() in obj1.label.lower())
                    tgt_match = (tgt_type == '*' or obj2.label.lower() == tgt_type.lower() or 
                                tgt_type.lower() in obj2.label.lower())
                    
                    # Match spatial relationship
                    spatial_match = (req_spatial == '*' or spatial_rel == req_spatial)
                    
                    if src_match and tgt_match and spatial_match:
                        # Calculate confidence
                        confidence = base_conf * spatial_conf
                        
                        # Create relationship
                        rel = Relationship(
                            id=next_rel_id,
                            type=func_rel,
                            source_id=source_id,
                            target_id=target_id,
                            confidence=confidence
                        )
                        
                        inferred_relationships.append(rel)
                        next_rel_id += 1
        
        # Add relationships to the graph
        for rel in inferred_relationships:
            self.add_relationship(rel)
            
        logger.info(f"Inferred {len(inferred_relationships)} functional relationships")
        
        return inferred_relationships
    
    def compute_scene_statistics(self):
        """Compute and update scene statistics."""
        logger.info("Computing scene statistics")
        
        if not self.objects:
            logger.warning("No objects in scene, skipping statistics")
            return
        
        # Calculate global bounding box
        all_points = []
        for obj in self.objects.values():
            all_points.extend(obj.points)
        
        all_points = np.array(all_points)
        if len(all_points) > 0:
            min_coords = np.min(all_points, axis=0)
            max_coords = np.max(all_points, axis=0)
            self.scene_info['global_bbox'] = np.concatenate([min_coords, max_coords])
            self.scene_info['dimensions'] = max_coords - min_coords
        
        # Count objects by type
        object_types = defaultdict(int)
        for obj in self.objects.values():
            object_types[obj.label] += 1
        
        # Count relationships by type
        relationship_types = defaultdict(int)
        for rel in self.relationships.values():
            relationship_types[rel.type] += 1
        
        # Compute graph statistics
        try:
            graph_stats = {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'connected_components': nx.number_connected_components(nx.Graph(self.graph)),
                'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes()),
            }
        except Exception as e:
            logger.warning(f"Error computing graph statistics: {e}")
            graph_stats = {
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
            }
        
        # Update statistics
        self.scene_info['stats'] = {
            'object_count': len(self.objects),
            'relationship_count': len(self.relationships),
            'object_types': dict(object_types),
            'relationship_types': dict(relationship_types),
            'graph': graph_stats,
        }
        
        logger.info(f"Scene has {len(self.objects)} objects and {len(self.relationships)} relationships")
    
    def save(self, filepath: str):
        """Save scene graph to file.
        
        Args:
            filepath: Path to save file
        """
        # Ensure directory exists
        ensure_dir(os.path.dirname(filepath))
        
        # Convert objects and relationships to dictionaries
        data = {
            'scene_info': self.scene_info,
            'objects': {obj_id: asdict(obj) for obj_id, obj in self.objects.items()},
            'relationships': {rel_id: asdict(rel) for rel_id, rel in self.relationships.items()},
        }
        
        # Convert numpy arrays to lists
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        data = convert_numpy(data)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved scene graph to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SceneGraph':
        """Load scene graph from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Loaded SceneGraph
        """
        # Load from file
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Create scene graph
        scene_graph = cls()
        
        # Set scene info
        scene_graph.scene_info = data['scene_info']
        
        # Convert dictionaries back to objects
        for obj_id, obj_dict in data['objects'].items():
            # Convert lists back to numpy arrays
            for key in ['points', 'center', 'bbox']:
                if key in obj_dict:
                    obj_dict[key] = np.array(obj_dict[key])
            
            # Create Object3D
            obj = Object3D(**obj_dict)
            scene_graph.add_object(obj)
        
        # Convert dictionaries back to relationships
        for rel_id, rel_dict in data['relationships'].items():
            # Create Relationship
            rel = Relationship(**rel_dict)
            scene_graph.add_relationship(rel)
        
        logger.info(f"Loaded scene graph from {filepath} with {len(scene_graph.objects)} objects "
                   f"and {len(scene_graph.relationships)} relationships")
        
        return scene_graph
    
    def visualize(self, output_path: Optional[str] = None):
        """Visualize scene graph using NetworkX and matplotlib.
        
        Args:
            output_path: Optional path to save visualization
        """
        logger.info("Visualizing scene graph")
        
        # Create a simplified graph for visualization
        G = nx.DiGraph()
        
        # Add nodes
        for obj_id, obj in self.objects.items():
            G.add_node(obj_id, label=obj.label)
        
        # Add edges
        for rel_id, rel in self.relationships.items():
            G.add_edge(rel.source_id, rel.target_id, 
                     label=rel.type, 
                     weight=rel.confidence)
        
        # Set up the plot
        plt.figure(figsize=(16, 12))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=42, k=0.15, iterations=100)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_size=700, 
                             node_color='lightblue',
                             alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, 
                             width=1.5, 
                             alpha=0.5, 
                             arrowsize=15)
        
        # Draw node labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, 
                              labels=node_labels,
                              font_size=10,
                              font_weight='bold')
        
        # Draw edge labels
        edge_labels = {(u, v): G.edges[u, v]['label'] 
                       for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, 
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        plt.title(f"Scene Graph: {self.scene_info['name']}", fontsize=16)
        plt.axis('off')
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved visualization to {output_path}")
        else:
            plt.tight_layout()
            plt.show()
    
    def extract_semantic_mesh(self, mesh: o3d.geometry.TriangleMesh) -> Dict[int, o3d.geometry.TriangleMesh]:
        """Extract separate meshes for each semantic object.
        
        Args:
            mesh: Input semantic mesh
            
        Returns:
            Dictionary mapping object IDs to meshes
        """
        # This is just a stub - actual implementation would extract submeshes
        # based on vertex labels and create a separate mesh for each object
        pass


def create_scene_graph_from_labeled_pointcloud(
        pointcloud: o3d.geometry.PointCloud,
        point_labels: np.ndarray, 
        metadata: List[Dict]) -> SceneGraph:
    """Create a scene graph from a labeled point cloud.
    
    Args:
        pointcloud: Labeled point cloud
        point_labels: Point labels (class IDs)
        metadata: Label metadata
        
    Returns:
        Scene graph
    """
    logger.info("Creating scene graph from labeled point cloud")
    
    # Create scene graph
    scene_graph = SceneGraph()
    
    # Set scene name
    scene_graph.scene_info['name'] = 'Point Cloud Scene'
    
    # Extract points
    points = np.asarray(pointcloud.points)
    
    # Group points by class
    points_by_class = defaultdict(list)
    indices_by_class = defaultdict(list)
    
    for i, label in enumerate(point_labels):
        if label >= 0:  # Skip unlabeled points
            points_by_class[label].append(points[i])
            indices_by_class[label].append(i)
    
    # Create objects
    next_obj_id = 0
    for class_id, class_points in points_by_class.items():
        # Get class metadata
        class_meta = next((m for m in metadata if m['class_id'] == class_id), None)
        if class_meta is None:
            logger.warning(f"No metadata found for class {class_id}, skipping")
            continue
        
        # Convert to numpy array
        class_points = np.array(class_points)
        
        # Compute bounding box
        min_coords = np.min(class_points, axis=0)
        max_coords = np.max(class_points, axis=0)
        bbox = np.concatenate([min_coords, max_coords])
        
        # Compute center
        center = (min_coords + max_coords) / 2
        
        # Create object
        obj = Object3D(
            id=next_obj_id,
            label=class_meta['class_name'],
            class_id=class_id,
            points=class_points,
            center=center,
            bbox=bbox,
            color=class_meta['color'],
            point_indices=indices_by_class[class_id]
        )
        
        # Add to scene graph
        scene_graph.add_object(obj)
        next_obj_id += 1
    
    logger.info(f"Created {len(scene_graph.objects)} objects")
    
    # Infer relationships
    scene_graph.infer_spatial_relationships()
    scene_graph.infer_functional_relationships()
    
    # Compute statistics
    scene_graph.compute_scene_statistics()
    
    return scene_graph


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and visualize scene graph")
    parser.add_argument("--input", required=True, help="Input labeled point cloud (.ply)")
    parser.add_argument("--labels", required=True, help="Point labels file (.npy)")
    parser.add_argument("--metadata", required=True, help="Label metadata file (.json)")
    parser.add_argument("--output", required=True, help="Output scene graph file (.pkl)")
    parser.add_argument("--visualize", action="store_true", help="Visualize scene graph")
    parser.add_argument("--vis-output", help="Path to save visualization")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load point cloud
    logger.info(f"Loading point cloud from {args.input}")
    pointcloud = o3d.io.read_point_cloud(args.input)
    
    # Load point labels
    logger.info(f"Loading point labels from {args.labels}")
    point_labels = np.load(args.labels)
    
    # Load metadata
    logger.info(f"Loading metadata from {args.metadata}")
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Create scene graph
    scene_graph = create_scene_graph_from_labeled_pointcloud(
        pointcloud, point_labels, metadata)
    
    # Save scene graph
    scene_graph.save(args.output)
    
    # Visualize if requested
    if args.visualize:
        vis_output = args.vis_output if args.vis_output else os.path.splitext(args.output)[0] + '.png'
        scene_graph.visualize(vis_output)


if __name__ == "__main__":
    main()