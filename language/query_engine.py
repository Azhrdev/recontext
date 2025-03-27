#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Natural language query engine for scene graphs.

This module enables natural language querying of 3D scene semantics,
translating questions into graph traversal operations.

Author: Lin Wei Sheng
Date: 2024-01-10
"""

import re
import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from tqdm import tqdm

# HuggingFace transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers import pipeline
    has_transformers = True
except ImportError:
    has_transformers = False
    logging.warning("transformers not installed. Install with: pip install transformers")

from recontext.language.scene_graph import SceneGraph, Object3D, Relationship
from recontext.language.graph_parser import GraphQuery, QueryResult, QueryParser
from recontext.utils.io_utils import download_model, ensure_dir

logger = logging.getLogger(__name__)

@dataclass
class NLQueryResult:
    """Results from a natural language query."""
    query: str  # Original query
    parsed_query: str  # Parsed structured query
    objects: List[int]  # List of object IDs matching query
    relationships: List[int]  # List of relationship IDs matching query
    answer: str  # Natural language answer
    confidence: float  # Confidence score
    details: Dict[str, Any] = None  # Additional details


class QueryEngine:
    """Natural language query engine for 3D scene graphs."""
    
    MODELS = {
        'default': {
            'name': 'recontext/scene-query-t5-base',
            'url': 'https://huggingface.co/t5-base',  # Placeholder URL
            'type': 'seq2seq'
        },
        'large': {
            'name': 'recontext/scene-query-t5-large',
            'url': 'https://huggingface.co/t5-large',  # Placeholder URL
            'type': 'seq2seq'
        }
    }
    
    def __init__(self, 
                 model_type: str = 'default',
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """Initialize the query engine.
        
        Args:
            model_type: Model type to use ('default' or 'large')
            device: Device to use ('cuda' or 'cpu')
            cache_dir: Directory to cache models
        """
        if not has_transformers:
            raise ImportError("transformers is required for QueryEngine")
        
        self.model_type = model_type
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".recontext", "models")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Initialize query parser
        self.parser = QueryParser()
        
        # Template responses
        self.templates = self._load_templates()
    
    def _init_models(self):
        """Initialize NLP models for query parsing and answering."""
        logger.info(f"Initializing query models of type: {self.model_type}")
        
        model_info = self.MODELS.get(self.model_type, self.MODELS['default'])
        model_name = model_info['name']
        
        # Ensure model directory exists
        ensure_dir(self.cache_dir)
        
        try:
            # Try to load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=self.cache_dir)
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Loaded model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_templates(self) -> Dict[str, str]:
        """Load template responses from file."""
        # Default templates
        templates = {
            'object_found': "I found {count} {object_type}. {details}",
            'object_not_found': "I couldn't find any {object_type} in the scene.",
            'relationship_found': "I found that {source} is {relation} {target}.",
            'relationship_not_found': "I couldn't find any relationship between {source} and {target}.",
            'location_answer': "The {object_type} is {relation} the {target}.",
            'count_answer': "There are {count} {object_type} in the scene.",
            'attribute_answer': "The {object_type} is {attribute}.",
            'fallback': "I'm not sure how to answer that question about the scene."
        }
        
        # Try to load custom templates if available
        template_path = os.path.join(os.path.dirname(__file__), 'templates.json')
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    custom_templates = json.load(f)
                templates.update(custom_templates)
                logger.info(f"Loaded {len(custom_templates)} custom templates")
            except Exception as e:
                logger.warning(f"Error loading custom templates: {e}")
        
        return templates
    
    def parse_query(self, query: str) -> str:
        """Parse natural language query into structured query format.
        
        Args:
            query: Natural language query
            
        Returns:
            Structured query string
        """
        # Normalize query
        query = query.lower().strip()
        if not query.endswith('?'):
            query += '?'
        
        # Use the model to parse the query
        inputs = self.tokenizer(f"translate to scene query: {query}", 
                               return_tensors="pt", 
                               max_length=512, 
                               truncation=True).to(self.device)
        
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )
        
        parsed_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sometimes the model adds prefixes like "scene query:" - remove them
        if ":" in parsed_query:
            parsed_query = parsed_query.split(":", 1)[1].strip()
        
        return parsed_query
    
    def execute_structured_query(self, 
                                scene_graph: SceneGraph, 
                                structured_query: str) -> QueryResult:
        """Execute a structured query on a scene graph.
        
        Args:
            scene_graph: Scene graph to query
            structured_query: Structured query string
            
        Returns:
            Query result
        """
        # Parse structured query
        graph_query = self.parser.parse(structured_query)
        
        # Execute query
        return self.parser.execute(graph_query, scene_graph)
    
    def query(self, scene_graph: SceneGraph, query: str) -> NLQueryResult:
        """Query the scene graph with natural language.
        
        Args:
            scene_graph: Scene graph to query
            query: Natural language query
            
        Returns:
            Query result
        """
        logger.info(f"Processing query: {query}")
        
        # Parse natural language query to structured query
        try:
            structured_query = self.parse_query(query)
            logger.info(f"Parsed query: {structured_query}")
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            structured_query = "PARSE_ERROR"
        
        # Execute structured query
        try:
            if structured_query != "PARSE_ERROR":
                query_result = self.execute_structured_query(scene_graph, structured_query)
                confidence = query_result.confidence
            else:
                query_result = QueryResult(
                    query_type="unknown",
                    object_ids=[],
                    relationship_ids=[],
                    attribute_values={},
                    confidence=0.2
                )
                confidence = 0.2
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            query_result = QueryResult(
                query_type="error",
                object_ids=[],
                relationship_ids=[],
                attribute_values={},
                confidence=0.1
            )
            confidence = 0.1
        
        # Generate natural language answer
        answer = self._generate_answer(scene_graph, query, structured_query, query_result)
        
        # Create result object
        result = NLQueryResult(
            query=query,
            parsed_query=structured_query,
            objects=query_result.object_ids,
            relationships=query_result.relationship_ids,
            answer=answer,
            confidence=confidence,
            details={
                'query_type': query_result.query_type,
                'attributes': query_result.attribute_values
            }
        )
        
        return result
    
    def _generate_answer(self, 
                         scene_graph: SceneGraph,
                         query: str,
                         structured_query: str,
                         query_result: QueryResult) -> str:
        """Generate natural language answer from query result.
        
        Args:
            scene_graph: Scene graph
            query: Original natural language query
            structured_query: Structured query
            query_result: Query result
            
        Returns:
            Natural language answer
        """
        # Handle error cases
        if structured_query == "PARSE_ERROR":
            return "I'm sorry, I couldn't understand that question about the scene."
        
        if query_result.query_type == "error":
            return "I encountered an error while trying to answer that question."
        
        # Extract query type
        query_type = query_result.query_type
        
        # Generate answer based on query type
        if query_type == "find":
            return self._generate_find_answer(scene_graph, query_result)
        elif query_type == "count":
            return self._generate_count_answer(scene_graph, query_result)
        elif query_type == "locate":
            return self._generate_location_answer(scene_graph, query_result)
        elif query_type == "attribute":
            return self._generate_attribute_answer(scene_graph, query_result)
        elif query_type == "relationship":
            return self._generate_relationship_answer(scene_graph, query_result)
        else:
            return self.templates['fallback']
    
    def _generate_find_answer(self, 
                              scene_graph: SceneGraph, 
                              query_result: QueryResult) -> str:
        """Generate answer for 'find' queries.
        
        Args:
            scene_graph: Scene graph
            query_result: Query result
            
        Returns:
            Natural language answer
        """
        object_ids = query_result.object_ids
        
        if not object_ids:
            # Try to extract object type from query
            object_type = query_result.details.get('object_type', 'objects')
            return self.templates['object_not_found'].format(object_type=object_type)
        
        # Get object types
        object_types = {}
        for obj_id in object_ids:
            obj = scene_graph.get_object(obj_id)
            if obj:
                object_types[obj.label] = object_types.get(obj.label, 0) + 1
        
        # Generate description
        if len(object_types) == 1:
            object_type = next(iter(object_types.keys()))
            count = object_types[object_type]
            
            if count == 1:
                details = f"It's located in the {self._describe_location(scene_graph, object_ids[0])}."
            else:
                details = f"They are scattered around the scene."
                
            return self.templates['object_found'].format(
                count=count,
                object_type=object_type,
                details=details
            )
        else:
            # Multiple object types
            type_descriptions = [f"{count} {obj_type}" 
                                for obj_type, count in object_types.items()]
            types_str = ", ".join(type_descriptions)
            
            return f"I found {types_str} in the scene."
    
    def _generate_count_answer(self, 
                               scene_graph: SceneGraph, 
                               query_result: QueryResult) -> str:
        """Generate answer for 'count' queries.
        
        Args:
            scene_graph: Scene graph
            query_result: Query result
            
        Returns:
            Natural language answer
        """
        count = len(query_result.object_ids)
        object_type = query_result.details.get('object_type', 'objects')
        
        return self.templates['count_answer'].format(
            count=count,
            object_type=object_type
        )
    
    def _generate_location_answer(self, 
                                 scene_graph: SceneGraph, 
                                 query_result: QueryResult) -> str:
        """Generate answer for 'locate' queries.
        
        Args:
            scene_graph: Scene graph
            query_result: Query result
            
        Returns:
            Natural language answer
        """
        object_ids = query_result.object_ids
        
        if not object_ids:
            object_type = query_result.details.get('object_type', 'object')
            return f"I couldn't find any {object_type} in the scene."
        
        # For simplicity, just describe the first object
        obj_id = object_ids[0]
        obj = scene_graph.get_object(obj_id)
        
        if not obj:
            return "I found the object but couldn't determine its location."
        
        location_desc = self._describe_location(scene_graph, obj_id)
        
        return f"The {obj.label} is {location_desc}."
    
    def _generate_attribute_answer(self, 
                                  scene_graph: SceneGraph, 
                                  query_result: QueryResult) -> str:
        """Generate answer for 'attribute' queries.
        
        Args:
            scene_graph: Scene graph
            query_result: Query result
            
        Returns:
            Natural language answer
        """
        attribute_values = query_result.attribute_values
        object_ids = query_result.object_ids
        
        if not object_ids or not attribute_values:
            return "I couldn't determine the attributes you're asking about."
        
        # For simplicity, just describe the first object
        obj_id = object_ids[0]
        obj = scene_graph.get_object(obj_id)
        
        if not obj:
            return "I couldn't find the object you're asking about."
        
        # Format attribute values
        attr_descriptions = []
        for attr, value in attribute_values.items():
            attr_descriptions.append(f"{attr}: {value}")
        
        attrs_str = ", ".join(attr_descriptions)
        
        return f"The {obj.label} has these attributes: {attrs_str}."
    
    def _generate_relationship_answer(self, 
                                     scene_graph: SceneGraph, 
                                     query_result: QueryResult) -> str:
        """Generate answer for 'relationship' queries.
        
        Args:
            scene_graph: Scene graph
            query_result: Query result
            
        Returns:
            Natural language answer
        """
        rel_ids = query_result.relationship_ids
        
        if not rel_ids:
            return "I couldn't find any relationships matching your query."
        
        # Describe each relationship
        descriptions = []
        for rel_id in rel_ids:
            rel = scene_graph.get_relationship(rel_id)
            if rel:
                source = scene_graph.get_object(rel.source_id)
                target = scene_graph.get_object(rel.target_id)
                
                if source and target:
                    descriptions.append(
                        f"the {source.label} is {rel.type} the {target.label}"
                    )
        
        if not descriptions:
            return "I found some relationships but couldn't describe them."
        
        if len(descriptions) == 1:
            return f"I found that {descriptions[0]}."
        else:
            return f"I found these relationships: {'; '.join(descriptions)}."
    
    def _describe_location(self, scene_graph: SceneGraph, obj_id: int) -> str:
        """Generate a location description for an object.
        
        Args:
            scene_graph: Scene graph
            obj_id: Object ID
            
        Returns:
            Location description
        """
        obj = scene_graph.get_object(obj_id)
        if not obj:
            return "unknown location"
        
        # Try to find relationships that describe location
        spatial_rels = []
        
        # Check outgoing relationships
        for rel_id, rel in scene_graph.relationships.items():
            if rel.source_id == obj_id and rel.type in scene_graph.SPATIAL_RELATIONSHIPS:
                target = scene_graph.get_object(rel.target_id)
                if target:
                    spatial_rels.append((rel.type, target.label, rel.confidence))
        
        # Check incoming relationships
        for rel_id, rel in scene_graph.relationships.items():
            if rel.target_id == obj_id and rel.type in scene_graph.SPATIAL_RELATIONSHIPS:
                source = scene_graph.get_object(rel.source_id)
                if source:
                    # Reverse the relationship type
                    rev_type = self._reverse_relationship(rel.type)
                    spatial_rels.append((rev_type, source.label, rel.confidence))
        
        # Sort by confidence
        spatial_rels.sort(key=lambda x: x[2], reverse=True)
        
        if spatial_rels:
            # Use the most confident relationship
            rel_type, other_obj, _ = spatial_rels[0]
            return f"{rel_type} the {other_obj}"
        else:
            # Fallback to using coordinates
            # Determine rough position (center, corner, etc.)
            scene_bbox = scene_graph.scene_info.get('global_bbox')
            if scene_bbox is not None:
                scene_center = (scene_bbox[:3] + scene_bbox[3:]) / 2
                pos = obj.center
                
                # Compare to scene center to determine rough position
                x_diff = pos[0] - scene_center[0]
                z_diff = pos[2] - scene_center[2]
                
                # Rough cardinal directions
                if abs(x_diff) > abs(z_diff):
                    if x_diff > 0:
                        return "on the right side of the scene"
                    else:
                        return "on the left side of the scene"
                else:
                    if z_diff > 0:
                        return "in the back of the scene"
                    else:
                        return "in the front of the scene"
            
            return "somewhere in the scene"
    
    def _reverse_relationship(self, rel_type: str) -> str:
        """Get the reverse of a relationship type.
        
        Args:
            rel_type: Relationship type
            
        Returns:
            Reversed relationship type
        """
        opposites = {
            'above': 'below',
            'below': 'above',
            'left_of': 'right_of',
            'right_of': 'left_of',
            'in_front_of': 'behind',
            'behind': 'in_front_of',
            'inside': 'contains',
            'contains': 'inside',
            'on_top_of': 'under',
            'under': 'on_top_of',
        }
        
        return opposites.get(rel_type, rel_type)


def main():
    """Main function for demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query a scene graph with natural language")
    parser.add_argument("--scene_graph", required=True, help="Path to scene graph (.pkl)")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--model", default="default", choices=["default", "large"],
                       help="Model type to use")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load scene graph
    logger.info(f"Loading scene graph from {args.scene_graph}")
    scene_graph = SceneGraph.load(args.scene_graph)
    
    # Create query engine
    engine = QueryEngine(model_type=args.model)
    
    # Process query
    result = engine.query(scene_graph, args.query)
    
    # Print results
    print(f"\nQuery: {result.query}")
    print(f"Parsed as: {result.parsed_query}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"\nAnswer: {result.answer}")
    
    if result.objects:
        print(f"\nMatching objects: {len(result.objects)}")
        for i, obj_id in enumerate(result.objects[:5]):  # Show at most 5
            obj = scene_graph.get_object(obj_id)
            if obj:
                print(f"  - {obj.label} (ID: {obj_id})")
        if len(result.objects) > 5:
            print(f"  ... and {len(result.objects) - 5} more")
    
    if result.relationships:
        print(f"\nMatching relationships: {len(result.relationships)}")
        for i, rel_id in enumerate(result.relationships[:5]):  # Show at most 5
            rel = scene_graph.get_relationship(rel_id)
            if rel:
                src = scene_graph.get_object(rel.source_id)
                tgt = scene_graph.get_object(rel.target_id)
                if src and tgt:
                    print(f"  - {src.label} {rel.type} {tgt.label} (ID: {rel_id})")
        if len(result.relationships) > 5:
            print(f"  ... and {len(result.relationships) - 5} more")


if __name__ == "__main__":
    main()