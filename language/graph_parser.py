#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph query parser for semantically rich scene graph queries.

This module translates structured query syntax into executable graph operations
on scene graphs, enabling semantic querying and reasoning about 3D scenes.

Author: Lin Wei Sheng
Date: 2024-01-08
Last modified: 2024-03-05
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass, field

from recontext.language.scene_graph import SceneGraph

logger = logging.getLogger(__name__)

@dataclass
class GraphQuery:
    """Represents a structured query for scene graphs."""
    query_type: str  # Type of query (find, count, locate, etc.)
    object_type: Optional[str] = None  # Type of object to query
    object_ids: List[int] = field(default_factory=list)  # Specific object IDs to query
    relationship_type: Optional[str] = None  # Type of relationship to query
    target_type: Optional[str] = None  # Target object type for relationship
    target_ids: List[int] = field(default_factory=list)  # Specific target IDs
    attribute: Optional[str] = None  # Attribute to query
    attribute_value: Optional[str] = None  # Value of attribute
    filters: Dict[str, Any] = field(default_factory=dict)  # Additional filters
    raw_query: str = ""  # Original raw query string

@dataclass
class QueryResult:
    """Results of a graph query execution."""
    query_type: str  # Type of query that was executed
    object_ids: List[int]  # Object IDs that match the query
    relationship_ids: List[int]  # Relationship IDs that match the query
    attribute_values: Dict[str, Any]  # Attribute values found
    confidence: float = 1.0  # Confidence score for the result
    details: Dict[str, Any] = field(default_factory=dict)  # Additional details

class QueryParser:
    """Parser for translating query strings into executable graph queries."""
    
    # Query type patterns
    FIND_PATTERN = r"find\s+(.+?)(?:\s+where\s+(.+))?\s*$"
    COUNT_PATTERN = r"count\s+(.+?)(?:\s+where\s+(.+))?\s*$"
    LOCATE_PATTERN = r"locate\s+(.+?)(?:\s+where\s+(.+))?\s*$"
    ATTRIBUTE_PATTERN = r"get\s+(.+?)\s+of\s+(.+?)(?:\s+where\s+(.+))?\s*$"
    RELATIONSHIP_PATTERN = r"relationship\s+(.+?)\s+between\s+(.+?)\s+and\s+(.+?)(?:\s+where\s+(.+))?\s*$"
    
    # Filter patterns
    FILTER_PATTERN = r"([^\s]+)\s+(=|!=|>|<|>=|<=|contains|startswith|endswith)\s+(.+)"
    
    def __init__(self):
        """Initialize the query parser."""
        pass
    
    def parse(self, query_str: str) -> GraphQuery:
        """Parse a query string into a structured GraphQuery.
        
        Args:
            query_str: The query string to parse
            
        Returns:
            Structured GraphQuery object
        """
        # Normalize query string
        query_str = query_str.strip().lower()
        
        # Try each query pattern
        query = None
        
        # Find queries
        match = re.match(self.FIND_PATTERN, query_str)
        if match:
            object_spec = match.group(1)
            filter_clause = match.group(2)
            
            query = GraphQuery(
                query_type="find",
                raw_query=query_str
            )
            
            self._parse_object_spec(query, object_spec)
            
            if filter_clause:
                self._parse_filters(query, filter_clause)
            
            return query
        
        # Count queries
        match = re.match(self.COUNT_PATTERN, query_str)
        if match:
            object_spec = match.group(1)
            filter_clause = match.group(2)
            
            query = GraphQuery(
                query_type="count",
                raw_query=query_str
            )
            
            self._parse_object_spec(query, object_spec)
            
            if filter_clause:
                self._parse_filters(query, filter_clause)
            
            return query
        
        # Locate queries
        match = re.match(self.LOCATE_PATTERN, query_str)
        if match:
            object_spec = match.group(1)
            filter_clause = match.group(2)
            
            query = GraphQuery(
                query_type="locate",
                raw_query=query_str
            )
            
            self._parse_object_spec(query, object_spec)
            
            if filter_clause:
                self._parse_filters(query, filter_clause)
            
            return query
        
        # Attribute queries
        match = re.match(self.ATTRIBUTE_PATTERN, query_str)
        if match:
            attribute = match.group(1)
            object_spec = match.group(2)
            filter_clause = match.group(3)
            
            query = GraphQuery(
                query_type="attribute",
                attribute=attribute,
                raw_query=query_str
            )
            
            self._parse_object_spec(query, object_spec)
            
            if filter_clause:
                self._parse_filters(query, filter_clause)
            
            return query
        
        # Relationship queries
        match = re.match(self.RELATIONSHIP_PATTERN, query_str)
        if match:
            relationship_type = match.group(1)
            source_spec = match.group(2)
            target_spec = match.group(3)
            filter_clause = match.group(4)
            
            query = GraphQuery(
                query_type="relationship",
                relationship_type=relationship_type,
                raw_query=query_str
            )
            
            self._parse_object_spec(query, source_spec)
            self._parse_target_spec(query, target_spec)
            
            if filter_clause:
                self._parse_filters(query, filter_clause)
            
            return query
        
        # If no pattern matched, create an unknown query
        logger.warning(f"Could not parse query: {query_str}")
        return GraphQuery(
            query_type="unknown",
            raw_query=query_str
        )
    
    def _parse_object_spec(self, query: GraphQuery, object_spec: str) -> None:
        """Parse object specification into query.
        
        Args:
            query: Query to update
            object_spec: Object specification string
        """
        # Check for specific IDs
        if object_spec.startswith("object") and "id" in object_spec:
            # Extract IDs (e.g., "object with id 5" or "objects with ids 1,2,3")
            id_match = re.search(r"id[s]?\s+(\d+(?:\s*,\s*\d+)*)", object_spec)
            if id_match:
                id_str = id_match.group(1)
                ids = [int(x.strip()) for x in id_str.split(",")]
                query.object_ids = ids
                return
        
        # Otherwise, it's an object type
        query.object_type = object_spec.strip()
    
    def _parse_target_spec(self, query: GraphQuery, target_spec: str) -> None:
        """Parse target specification into query.
        
        Args:
            query: Query to update
            target_spec: Target specification string
        """
        # Check for specific IDs
        if target_spec.startswith("object") and "id" in target_spec:
            # Extract IDs
            id_match = re.search(r"id[s]?\s+(\d+(?:\s*,\s*\d+)*)", target_spec)
            if id_match:
                id_str = id_match.group(1)
                ids = [int(x.strip()) for x in id_str.split(",")]
                query.target_ids = ids
                return
        
        # Otherwise, it's an object type
        query.target_type = target_spec.strip()
    
    def _parse_filters(self, query: GraphQuery, filter_clause: str) -> None:
        """Parse filter clause into query filters.
        
        Args:
            query: Query to update
            filter_clause: Filter clause string
        """
        # Split multiple filters (separated by 'and')
        filter_parts = filter_clause.split(" and ")
        
        for part in filter_parts:
            match = re.match(self.FILTER_PATTERN, part.strip())
            if match:
                attribute = match.group(1)
                operator = match.group(2)
                value = match.group(3)
                
                # Handle quoted values
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                # Handle numeric values
                elif value.isdigit():
                    value = int(value)
                elif re.match(r"^\d+\.\d+$", value):
                    value = float(value)
                
                # Store filter
                query.filters[attribute] = {
                    "operator": operator,
                    "value": value
                }
    
    def execute(self, query: GraphQuery, scene_graph: SceneGraph) -> QueryResult:
        """Execute a graph query on a scene graph.
        
        Args:
            query: The query to execute
            scene_graph: The scene graph to query
            
        Returns:
            Query result
        """
        # Check query type
        if query.query_type == "find":
            return self._execute_find(query, scene_graph)
        elif query.query_type == "count":
            return self._execute_count(query, scene_graph)
        elif query.query_type == "locate":
            return self._execute_locate(query, scene_graph)
        elif query.query_type == "attribute":
            return self._execute_attribute(query, scene_graph)
        elif query.query_type == "relationship":
            return self._execute_relationship(query, scene_graph)
        else:
            logger.warning(f"Unknown query type: {query.query_type}")
            return QueryResult(
                query_type="unknown",
                object_ids=[],
                relationship_ids=[],
                attribute_values={},
                confidence=0.1
            )
    
    def _execute_find(self, query: GraphQuery, scene_graph: SceneGraph) -> QueryResult:
        """Execute a find query.
        
        Args:
            query: Find query
            scene_graph: Scene graph
            
        Returns:
            Query result
        """
        # First, find matching objects
        objects = []
        
        # If specific IDs are given, use those
        if query.object_ids:
            objects = [scene_graph.get_object(obj_id) for obj_id in query.object_ids 
                     if scene_graph.get_object(obj_id) is not None]
        # Otherwise, find by type
        elif query.object_type:
            objects = scene_graph.get_objects_by_label(query.object_type)
        # If neither is specified, get all objects
        else:
            objects = list(scene_graph.objects.values())
        
        # Apply filters
        filtered_objects = self._apply_filters(objects, query.filters)
        
        # Extract object IDs
        object_ids = [obj.id for obj in filtered_objects]
        
        # Get confidence based on how specific the query was
        confidence = 0.9 if query.object_ids or query.object_type else 0.7
        
        # Create result
        result = QueryResult(
            query_type="find",
            object_ids=object_ids,
            relationship_ids=[],
            attribute_values={},
            confidence=confidence,
            details={"object_type": query.object_type}
        )
        
        return result
    
    def _execute_count(self, query: GraphQuery, scene_graph: SceneGraph) -> QueryResult:
        """Execute a count query.
        
        Args:
            query: Count query
            scene_graph: Scene graph
            
        Returns:
            Query result
        """
        # Execute as a find query first
        find_result = self._execute_find(query, scene_graph)
        
        # Return the count result
        return QueryResult(
            query_type="count",
            object_ids=find_result.object_ids,
            relationship_ids=[],
            attribute_values={"count": len(find_result.object_ids)},
            confidence=find_result.confidence,
            details={"object_type": query.object_type}
        )
    
    def _execute_locate(self, query: GraphQuery, scene_graph: SceneGraph) -> QueryResult:
        """Execute a locate query.
        
        Args:
            query: Locate query
            scene_graph: Scene graph
            
        Returns:
            Query result
        """
        # Execute as a find query first
        find_result = self._execute_find(query, scene_graph)
        
        # Find spatial relationships for the objects
        relationship_ids = []
        
        for obj_id in find_result.object_ids:
            # Get object
            obj = scene_graph.get_object(obj_id)
            if not obj:
                continue
                
            # Find spatial relationships
            for rel_id, rel in scene_graph.relationships.items():
                if rel.type in scene_graph.SPATIAL_RELATIONSHIPS:
                    if rel.source_id == obj_id or rel.target_id == obj_id:
                        relationship_ids.append(rel_id)
        
        # Return the locate result
        return QueryResult(
            query_type="locate",
            object_ids=find_result.object_ids,
            relationship_ids=relationship_ids,
            attribute_values={},
            confidence=find_result.confidence,
            details={"object_type": query.object_type}
        )
    
    def _execute_attribute(self, query: GraphQuery, scene_graph: SceneGraph) -> QueryResult:
        """Execute an attribute query.
        
        Args:
            query: Attribute query
            scene_graph: Scene graph
            
        Returns:
            Query result
        """
        # Execute as a find query first to get the objects
        find_result = self._execute_find(query, scene_graph)
        
        # Get attribute values for the objects
        attribute_values = {}
        
        for obj_id in find_result.object_ids:
            # Get object
            obj = scene_graph.get_object(obj_id)
            if not obj:
                continue
                
            # Get attribute value based on query attribute
            if query.attribute in obj.attributes:
                # Attribute exists in object
                attribute_values[obj_id] = obj.attributes[query.attribute]
            elif query.attribute == "color":
                attribute_values[obj_id] = obj.color
            elif query.attribute == "position" or query.attribute == "center":
                attribute_values[obj_id] = obj.center
            elif query.attribute == "size":
                attribute_values[obj_id] = obj.size
            elif query.attribute == "volume":
                attribute_values[obj_id] = obj.volume
            elif query.attribute == "bbox" or query.attribute == "bounding_box":
                attribute_values[obj_id] = obj.bbox
        
        # Determine confidence
        confidence = find_result.confidence * (0.9 if attribute_values else 0.5)
        
        # Return the attribute result
        return QueryResult(
            query_type="attribute",
            object_ids=find_result.object_ids,
            relationship_ids=[],
            attribute_values=attribute_values,
            confidence=confidence,
            details={
                "object_type": query.object_type,
                "attribute": query.attribute
            }
        )
    
    def _execute_relationship(self, query: GraphQuery, scene_graph: SceneGraph) -> QueryResult:
        """Execute a relationship query.
        
        Args:
            query: Relationship query
            scene_graph: Scene graph
            
        Returns:
            Query result
        """
        # Find source objects
        source_objects = []
        if query.object_ids:
            source_objects = [scene_graph.get_object(obj_id) for obj_id in query.object_ids 
                           if scene_graph.get_object(obj_id) is not None]
        elif query.object_type:
            source_objects = scene_graph.get_objects_by_label(query.object_type)
        else:
            source_objects = list(scene_graph.objects.values())
        
        # Find target objects
        target_objects = []
        if query.target_ids:
            target_objects = [scene_graph.get_object(obj_id) for obj_id in query.target_ids 
                            if scene_graph.get_object(obj_id) is not None]
        elif query.target_type:
            target_objects = scene_graph.get_objects_by_label(query.target_type)
        else:
            target_objects = list(scene_graph.objects.values())
        
        # Apply filters to source and target objects
        source_objects = self._apply_filters(source_objects, query.filters)
        target_objects = self._apply_filters(target_objects, query.filters)
        
        # Find relationships between source and target objects
        relationship_ids = []
        
        for source in source_objects:
            for target in target_objects:
                # Skip self-relationships
                if source.id == target.id:
                    continue
                
                # Find relationships between source and target
                for rel_id, rel in scene_graph.relationships.items():
                    if rel.source_id == source.id and rel.target_id == target.id:
                        # Check if relationship type matches
                        if query.relationship_type is None or rel.type == query.relationship_type:
                            relationship_ids.append(rel_id)
        
        # Extract object IDs
        source_ids = [obj.id for obj in source_objects]
        target_ids = [obj.id for obj in target_objects]
        
        # Determine confidence
        has_rel_type = query.relationship_type is not None
        confidence = 0.8 if has_rel_type else 0.6
        
        # Return the relationship result
        return QueryResult(
            query_type="relationship",
            object_ids=list(set(source_ids + target_ids)),  # Combine and deduplicate
            relationship_ids=relationship_ids,
            attribute_values={},
            confidence=confidence,
            details={
                "source_type": query.object_type,
                "target_type": query.target_type,
                "relationship_type": query.relationship_type
            }
        )
    
    def _apply_filters(self, objects: List[Any], filters: Dict[str, Any]) -> List[Any]:
        """Apply filters to a list of objects.
        
        Args:
            objects: List of objects to filter
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered list of objects
        """
        if not filters:
            return objects
        
        filtered_objects = []
        
        for obj in objects:
            # Check if object passes all filters
            passes_all = True
            
            for attr, filter_spec in filters.items():
                operator = filter_spec["operator"]
                value = filter_spec["value"]
                
                # Get object attribute value
                if attr in obj.attributes:
                    obj_value = obj.attributes[attr]
                elif hasattr(obj, attr):
                    obj_value = getattr(obj, attr)
                else:
                    # Attribute not found, filter fails
                    passes_all = False
                    break
                
                # Apply operator
                if operator == "=":
                    passes = obj_value == value
                elif operator == "!=":
                    passes = obj_value != value
                elif operator == ">":
                    passes = obj_value > value
                elif operator == "<":
                    passes = obj_value < value
                elif operator == ">=":
                    passes = obj_value >= value
                elif operator == "<=":
                    passes = obj_value <= value
                elif operator == "contains":
                    if isinstance(obj_value, str) and isinstance(value, str):
                        passes = value in obj_value
                    elif hasattr(obj_value, "__contains__"):
                        passes = value in obj_value
                    else:
                        passes = False
                elif operator == "startswith":
                    if isinstance(obj_value, str) and isinstance(value, str):
                        passes = obj_value.startswith(value)
                    else:
                        passes = False
                elif operator == "endswith":
                    if isinstance(obj_value, str) and isinstance(value, str):
                        passes = obj_value.endswith(value)
                    else:
                        passes = False
                else:
                    # Unknown operator, filter fails
                    passes = False
                
                if not passes:
                    passes_all = False
                    break
            
            if passes_all:
                filtered_objects.append(obj)
        
        return filtered_objects