#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Natural language interface for 3D scene understanding.

This module provides interfaces for converting natural language queries into
structured graph queries and generating natural language responses.

Author: Lin Wei Sheng
Date: 2024-01-15
Last modified: 2024-03-10
"""

import re
import os
import json
import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict

from recontext.language.scene_graph import SceneGraph
from recontext.language.graph_parser import GraphQuery, QueryResult

logger = logging.getLogger(__name__)

# Try to import spaCy
try:
    import spacy
    has_spacy = True
except ImportError:
    has_spacy = False
    logger.warning("spaCy not installed. Install with: pip install spacy")
    logger.warning("You'll also need to download a model: python -m spacy download en_core_web_sm")

# Try to import HuggingFace transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    has_transformers = True
except ImportError:
    has_transformers = False
    logger.warning("transformers not installed. Install with: pip install transformers")


class NaturalLanguageProcessor:
    """Natural language processing tools for scene understanding queries."""
    
    def __init__(self, 
                 nlp_model: str = "en_core_web_sm",
                 use_transformers: bool = True,
                 device: Optional[str] = None):
        """Initialize natural language processor.
        
        Args:
            nlp_model: spaCy model to use
            use_transformers: Whether to use HuggingFace transformers
            device: Device to use for transformers ('cuda' or 'cpu')
        """
        self.use_transformers = use_transformers and has_transformers
        
        # Load spaCy model if available
        if has_spacy:
            try:
                self.nlp = spacy.load(nlp_model)
                logger.info(f"Loaded spaCy model: {nlp_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.nlp = None
        else:
            self.nlp = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load transformers model if available
        if self.use_transformers:
            self._init_transformers()
        
        # Load response templates
        self.templates = self._load_templates()
    
    def _init_transformers(self):
        """Initialize transformers models for query parsing and generation."""
        try:
            # Load query parsing model
            self.query_tokenizer = AutoTokenizer.from_pretrained(
                "t5-base", 
                model_max_length=512
            )
            self.query_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(self.device)
            
            # Load response generation model
            self.response_tokenizer = AutoTokenizer.from_pretrained(
                "t5-base", 
                model_max_length=512
            )
            self.response_model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(self.device)
            
            logger.info("Loaded transformers models")
            
        except Exception as e:
            logger.warning(f"Failed to load transformers models: {e}")
            self.use_transformers = False
    
    def _load_templates(self) -> Dict[str, str]:
        """Load response templates from file."""
        # Default templates
        templates = {
            'find': "I found {count} {object_type} in the scene.",
            'count': "There are {count} {object_type} in the scene.",
            'locate': "The {object_type} is {location}.",
            'attribute': "The {attribute} of the {object_type} is {value}.",
            'relationship': "The {source} is {relationship} the {target}.",
            'not_found': "I couldn't find any {object_type} in the scene.",
            'error': "I'm sorry, I couldn't understand that query.",
            'unknown': "I'm not sure how to answer that question about the scene."
        }
        
        # Try to load custom templates
        template_path = os.path.join(os.path.dirname(__file__), 'templates.json')
        if os.path.exists(template_path):
            try:
                with open(template_path, 'r') as f:
                    custom_templates = json.load(f)
                templates.update(custom_templates)
                logger.info(f"Loaded {len(custom_templates)} custom templates")
            except Exception as e:
                logger.warning(f"Failed to load custom templates: {e}")
        
        return templates
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent and entities from a natural language query.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary of intent and entities
        """
        # Initialize intent
        intent = {
            'action': 'unknown',
            'object_type': None,
            'attributes': [],
            'relationships': [],
            'location': None,
            'confidence': 0.0
        }
        
        # Use spaCy for intent recognition if available
        if self.nlp is not None:
            doc = self.nlp(query)
            
            # Extract verb (action)
            action_verbs = {
                'find': ['find', 'show', 'display', 'identify', 'locate'],
                'count': ['count', 'how many', 'number of'],
                'locate': ['where', 'position', 'location', 'place'],
                'attribute': ['what is', 'describe', 'properties', 'characteristics'],
                'relationship': ['relation', 'connected', 'linked', 'between']
            }
            
            # Extract verb
            main_verb = None
            for token in doc:
                if token.pos_ == "VERB":
                    main_verb = token.lemma_
                    break
            
            # Determine action
            if main_verb:
                for action, verbs in action_verbs.items():
                    if main_verb in verbs or any(v in query.lower() for v in verbs):
                        intent['action'] = action
                        intent['confidence'] = 0.7
                        break
            
            # Extract object types (nouns)
            for chunk in doc.noun_chunks:
                if chunk.root.dep_ in ('nsubj', 'dobj', 'pobj'):
                    intent['object_type'] = chunk.text
                    intent['confidence'] = max(intent['confidence'], 0.8)
                    break
            
            # Extract attributes
            for token in doc:
                if token.pos_ == "ADJ":
                    intent['attributes'].append(token.text)
                    intent['confidence'] = max(intent['confidence'], 0.6)
            
            # Extract location
            location_preps = ['in', 'on', 'at', 'near', 'by', 'inside', 'outside', 'above', 'below']
            for token in doc:
                if token.text in location_preps and token.head.pos_ == "NOUN":
                    intent['location'] = token.head.text
                    intent['confidence'] = max(intent['confidence'], 0.7)
        
        # Use transformers for more complex intent if available
        elif self.use_transformers:
            input_text = f"Extract intent from: {query}"
            
            # Tokenize and generate
            inputs = self.query_tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.query_model.generate(
                inputs['input_ids'],
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode and parse
            output_text = self.query_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract intent
            if "find" in output_text.lower():
                intent['action'] = 'find'
                intent['confidence'] = 0.7
            elif "count" in output_text.lower():
                intent['action'] = 'count'
                intent['confidence'] = 0.7
            elif "locate" in output_text.lower() or "where" in output_text.lower():
                intent['action'] = 'locate'
                intent['confidence'] = 0.7
            elif "attribute" in output_text.lower() or "property" in output_text.lower():
                intent['action'] = 'attribute'
                intent['confidence'] = 0.7
            elif "relation" in output_text.lower():
                intent['action'] = 'relationship'
                intent['confidence'] = 0.7
            
            # Extract object type (simple regex)
            obj_match = re.search(r'object[_\s]type[:\s]+([^\s,\.]+)', output_text)
            if obj_match:
                intent['object_type'] = obj_match.group(1)
                intent['confidence'] = max(intent['confidence'], 0.8)
        
        # Fallback to simple keyword matching
        else:
            # Simple keyword matching
            query_lower = query.lower()
            
            if "how many" in query_lower or "count" in query_lower:
                intent['action'] = 'count'
                intent['confidence'] = 0.6
            elif "where" in query_lower:
                intent['action'] = 'locate'
                intent['confidence'] = 0.6
            elif "what is" in query_lower or "describe" in query_lower:
                intent['action'] = 'attribute'
                intent['confidence'] = 0.6
            elif "relation" in query_lower or "between" in query_lower:
                intent['action'] = 'relationship'
                intent['confidence'] = 0.6
            else:
                intent['action'] = 'find'
                intent['confidence'] = 0.5
            
            # Try to extract object type using regex
            obj_match = re.search(r'(table|chair|sofa|bed|lamp|book|cup|bottle|tv|computer|desk|shelf|cabinet|door|window)', query_lower)
            if obj_match:
                intent['object_type'] = obj_match.group(1)
                intent['confidence'] = max(intent['confidence'], 0.7)
        
        return intent
    
    def nl_to_structured_query(self, query: str) -> str:
        """Convert natural language query to structured query format.
        
        Args:
            query: Natural language query
            
        Returns:
            Structured query string
        """
        # Extract intent
        intent = self.extract_query_intent(query)
        
        # Build structured query based on intent
        if intent['action'] == 'find':
            if intent['object_type']:
                struct_query = f"find {intent['object_type']}"
            else:
                struct_query = "find objects"
        
        elif intent['action'] == 'count':
            if intent['object_type']:
                struct_query = f"count {intent['object_type']}"
            else:
                struct_query = "count objects"
        
        elif intent['action'] == 'locate':
            if intent['object_type']:
                struct_query = f"locate {intent['object_type']}"
            else:
                struct_query = "locate objects"
        
        elif intent['action'] == 'attribute':
            if intent['attributes'] and intent['object_type']:
                struct_query = f"get {intent['attributes'][0]} of {intent['object_type']}"
            elif intent['object_type']:
                struct_query = f"get properties of {intent['object_type']}"
            else:
                struct_query = "get properties of objects"
        
        elif intent['action'] == 'relationship':
            if intent['object_type'] and intent['location']:
                struct_query = f"relationship * between {intent['object_type']} and {intent['location']}"
            else:
                struct_query = "relationship * between objects"
        
        else:
            # Fallback to a simple find query
            struct_query = "find objects"
        
        # Add filters if location is specified
        if intent['location'] and intent['action'] not in ['locate', 'relationship']:
            struct_query += f" where location = {intent['location']}"
        
        # Use transformers for more complex queries if available
        if self.use_transformers:
            try:
                input_text = f"Translate to scene query: {query}"
                
                # Tokenize and generate
                inputs = self.query_tokenizer(input_text, return_tensors="pt").to(self.device)
                outputs = self.query_model.generate(
                    inputs['input_ids'],
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
                
                # Decode
                trans_query = self.query_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Check if translation makes sense
                if trans_query and any(x in trans_query for x in ["find", "count", "locate", "get", "relationship"]):
                    return trans_query
            except Exception as e:
                logger.warning(f"Failed to translate query with transformers: {e}")
        
        return struct_query
    
    def generate_response(self, 
                         query_result: QueryResult, 
                         scene_graph: SceneGraph) -> str:
        """Generate natural language response from query result.
        
        Args:
            query_result: Result of graph query
            scene_graph: Scene graph
            
        Returns:
            Natural language response
        """
        # Extract query type
        query_type = query_result.query_type
        
        # Generate response based on query type
        if query_type == "find":
            return self._generate_find_response(query_result, scene_graph)
        elif query_type == "count":
            return self._generate_count_response(query_result, scene_graph)
        elif query_type == "locate":
            return self._generate_locate_response(query_result, scene_graph)
        elif query_type == "attribute":
            return self._generate_attribute_response(query_result, scene_graph)
        elif query_type == "relationship":
            return self._generate_relationship_response(query_result, scene_graph)
        else:
            return self.templates['unknown']
    
    def _generate_find_response(self, 
                               query_result: QueryResult, 
                               scene_graph: SceneGraph) -> str:
        """Generate response for find queries.
        
        Args:
            query_result: Result of find query
            scene_graph: Scene graph
            
        Returns:
            Natural language response
        """
        object_ids = query_result.object_ids
        
        if not object_ids:
            object_type = query_result.details.get('object_type', 'objects')
            return self.templates['not_found'].format(object_type=object_type)
        
        # Get object types
        object_types = {}
        for obj_id in object_ids:
            obj = scene_graph.get_object(obj_id)
            if obj:
                object_types[obj.label] = object_types.get(obj.label, 0) + 1
        
        # Generate response
        if len(object_types) == 1:
            object_type = next(iter(object_types.keys()))
            count = object_types[object_type]
            
            response = self.templates['find'].format(
                count=count,
                object_type=object_type if count == 1 else object_type + 's'
            )
        else:
            # Multiple object types
            type_counts = [f"{count} {obj_type}" for obj_type, count in object_types.items()]
            types_str = ", ".join(type_counts)
            
            response = f"I found {types_str} in the scene."
        
        return response
    
    def _generate_count_response(self, 
                                query_result: QueryResult, 
                                scene_graph: SceneGraph) -> str:
        """Generate response for count queries.
        
        Args:
            query_result: Result of count query
            scene_graph: Scene graph
            
        Returns:
            Natural language response
        """
        count = len(query_result.object_ids)
        object_type = query_result.details.get('object_type', 'objects')
        
        if object_type.endswith('s') and count == 1:
            # Make singular
            object_type = object_type[:-1]
        elif not object_type.endswith('s') and count != 1:
            # Make plural
            object_type = object_type + 's'
        
        return self.templates['count'].format(
            count=count,
            object_type=object_type
        )
    
    def _generate_locate_response(self, 
                                 query_result: QueryResult, 
                                 scene_graph: SceneGraph) -> str:
        """Generate response for locate queries.
        
        Args:
            query_result: Result of locate query
            scene_graph: Scene graph
            
        Returns:
            Natural language response
        """
        object_ids = query_result.object_ids
        
        if not object_ids:
            object_type = query_result.details.get('object_type', 'object')
            return self.templates['not_found'].format(object_type=object_type)
        
        # For simplicity, just describe the first object
        obj_id = object_ids[0]
        obj = scene_graph.get_object(obj_id)
        
        if not obj:
            return "I found the object but couldn't determine its location."
        
        # Get location description
        location = self._describe_location(obj_id, scene_graph)
        
        return self.templates['locate'].format(
            object_type=obj.label,
            location=location
        )
    
    def _generate_attribute_response(self, 
                                    query_result: QueryResult, 
                                    scene_graph: SceneGraph) -> str:
        """Generate response for attribute queries.
        
        Args:
            query_result: Result of attribute query
            scene_graph: Scene graph
            
        Returns:
            Natural language response
        """
        attribute_values = query_result.attribute_values
        object_ids = query_result.object_ids
        attribute = query_result.details.get('attribute', 'property')
        
        if not object_ids or not attribute_values:
            object_type = query_result.details.get('object_type', 'object')
            return f"I couldn't determine the {attribute} of the {object_type}."
        
        # For simplicity, just describe the first object
        obj_id = object_ids[0]
        obj = scene_graph.get_object(obj_id)
        
        if not obj:
            return f"I couldn't find the object you're asking about."
        
        # Get attribute value
        value = attribute_values.get(obj_id, "unknown")
        
        # Format value
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) <= 3:  # For colors, positions, etc.
                value = "[" + ", ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in value) + "]"
            else:
                value = f"[{len(value)} values]"
        
        return self.templates['attribute'].format(
            attribute=attribute,
            object_type=obj.label,
            value=value
        )
    
    def _generate_relationship_response(self, 
                                       query_result: QueryResult, 
                                       scene_graph: SceneGraph) -> str:
        """Generate response for relationship queries.
        
        Args:
            query_result: Result of relationship query
            scene_graph: Scene graph
            
        Returns:
            Natural language response
        """
        rel_ids = query_result.relationship_ids
        
        if not rel_ids:
            source_type = query_result.details.get('source_type', 'object')
            target_type = query_result.details.get('target_type', 'object')
            return f"I couldn't find any relationship between {source_type} and {target_type}."
        
        # Describe each relationship
        descriptions = []
        for rel_id in rel_ids[:3]:  # Limit to 3 for readability
            rel = scene_graph.get_relationship(rel_id)
            if rel:
                source = scene_graph.get_object(rel.source_id)
                target = scene_graph.get_object(rel.target_id)
                
                if source and target:
                    descriptions.append(
                        self.templates['relationship'].format(
                            source=source.label,
                            relationship=rel.type,
                            target=target.label
                        )
                    )
        
        if not descriptions:
            return "I found some relationships but couldn't describe them."
        
        if len(rel_ids) > 3:
            descriptions.append(f"... and {len(rel_ids) - 3} more relationships.")
        
        return " ".join(descriptions)
    
    def _describe_location(self, obj_id: int, scene_graph: SceneGraph) -> str:
        """Generate a location description for an object.
        
        Args:
            obj_id: Object ID
            scene_graph: Scene graph
            
        Returns:
            Location description
        """
        obj = scene_graph.get_object(obj_id)
        if not obj:
            return "in an unknown location"
        
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
        
        # Fallback to coordinates
        return "in the scene"
    
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


class NLDialogueManager:
    """Manages dialogue interaction for natural language scene queries."""
    
    def __init__(self, 
                 scene_graph: SceneGraph,
                 nlp_processor: Optional[NaturalLanguageProcessor] = None):
        """Initialize dialogue manager.
        
        Args:
            scene_graph: Scene graph for the 3D scene
            nlp_processor: Natural language processor (creates new one if None)
        """
        self.scene_graph = scene_graph
        
        # Create NLP processor if not provided
        if nlp_processor is None:
            self.nlp = NaturalLanguageProcessor()
        else:
            self.nlp = nlp_processor
        
        # Create graph parser
        from recontext.language.graph_parser import QueryParser
        self.parser = QueryParser()
        
        # Dialogue history
        self.dialogue_history = []
        
        # Last context objects (for resolving references)
        self.last_objects = []
    
    def process_query(self, query: str) -> str:
        """Process natural language query and generate response.
        
        Args:
            query: Natural language query
            
        Returns:
            Natural language response
        """
        # Add query to history
        self.dialogue_history.append({"role": "user", "text": query})
        
        # Resolve references (e.g., "it", "them")
        resolved_query = self._resolve_references(query)
        
        # Convert to structured query
        structured_query = self.nlp.nl_to_structured_query(resolved_query)
        
        # Parse structured query
        graph_query = self.parser.parse(structured_query)
        
        # Execute query
        result = self.parser.execute(graph_query, self.scene_graph)
        
        # Update context objects
        if result.object_ids:
            self.last_objects = result.object_ids
        
        # Generate response
        response = self.nlp.generate_response(result, self.scene_graph)
        
        # Add response to history
        self.dialogue_history.append({"role": "system", "text": response})
        
        return response
    
    def _resolve_references(self, query: str) -> str:
        """Resolve references in a query using dialogue history.
        
        Args:
            query: Natural language query
            
        Returns:
            Query with references resolved
        """
        # Skip if no history or no references
        if not self.dialogue_history or not self.last_objects:
            return query
        
        # Check for references
        query_lower = query.lower()
        reference_words = ['it', 'this', 'that', 'they', 'them', 'these', 'those']
        
        has_reference = any(ref in query_lower.split() for ref in reference_words)
        
        if not has_reference:
            return query
        
        # Get object types for replacement
        object_types = []
        for obj_id in self.last_objects:
            obj = self.scene_graph.get_object(obj_id)
            if obj and obj.label not in object_types:
                object_types.append(obj.label)
        
        if not object_types:
            return query
        
        # Create replacement string
        if len(object_types) == 1:
            replacement = f"the {object_types[0]}"
        else:
            replacement = f"the {' and '.join(object_types)}"
        
        # Replace references
        resolved_query = query
        for ref in reference_words:
            # Replace whole words only
            pattern = r'\b' + re.escape(ref) + r'\b'
            resolved_query = re.sub(pattern, replacement, resolved_query, flags=re.IGNORECASE)
        
        return resolved_query