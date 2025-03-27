#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for querying a 3D scene using natural language.

This demonstrates how to use the RECONTEXT query engine to ask questions
about a 3D scene and visualize the results.

Author: Lin Wei Sheng
Date: 2024-02-15
Last modified: 2024-03-15
"""

import os
import sys
import argparse
import logging
import numpy as np
import open3d as o3d
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recontext.language.scene_graph import SceneGraph
from recontext.language.query_engine import QueryEngine
from recontext.visualization.interactive_viewer import InteractiveViewer
from recontext.utils.io_utils import ensure_dir

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RECONTEXT Scene Query Example")
    
    parser.add_argument("--scene_graph", required=True, help="Path to scene graph file (.pkl)")
    parser.add_argument("--pointcloud", help="Path to labeled point cloud file (.ply)")
    parser.add_argument("--mesh", help="Path to labeled mesh file (.ply)")
    parser.add_argument("--query", help="Natural language query (if not provided, interactive mode is used)")
    parser.add_argument("--model_type", choices=["default", "large"], default="default",
                       help="Query engine model type")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive viewer")
    
    return parser.parse_args()

def print_result(result):
    """Print query result in a formatted way."""
    print("\n" + "="*50)
    print(f"Query: {result.query}")
    print(f"Parsed as: {result.parsed_query}")
    print("-"*50)
    print(f"Answer: {result.answer}")
    print("-"*50)
    
    if result.objects:
        print(f"Matching Objects ({len(result.objects)}):")
        for i, obj_id in enumerate(result.objects[:5]):  # Show up to 5 objects
            print(f"  Object ID: {obj_id}")
        if len(result.objects) > 5:
            print(f"  ... and {len(result.objects) - 5} more")
    
    if result.relationships:
        print(f"Matching Relationships ({len(result.relationships)}):")
        for i, rel_id in enumerate(result.relationships[:5]):  # Show up to 5 relationships
            print(f"  Relationship ID: {rel_id}")
        if len(result.relationships) > 5:
            print(f"  ... and {len(result.relationships) - 5} more")
    
    print(f"Confidence: {result.confidence:.2f}")
    print("="*50 + "\n")

def save_result(result, scene_graph, output_dir):
    """Save query result to output directory."""
    # Create output directory
    ensure_dir(output_dir)
    
    # Save query and answer
    with open(os.path.join(output_dir, "query_result.txt"), "w") as f:
        f.write(f"Query: {result.query}\n")
        f.write(f"Parsed Query: {result.parsed_query}\n")
        f.write(f"Answer: {result.answer}\n")
        f.write(f"Confidence: {result.confidence:.2f}\n\n")
        
        if result.objects:
            f.write(f"Matching Objects ({len(result.objects)}):\n")
            for obj_id in result.objects:
                obj = scene_graph.get_object(obj_id)
                if obj:
                    f.write(f"  Object ID: {obj_id}, Label: {obj.label}\n")
        
        if result.relationships:
            f.write(f"Matching Relationships ({len(result.relationships)}):\n")
            for rel_id in result.relationships:
                rel = scene_graph.get_relationship(rel_id)
                if rel:
                    source = scene_graph.get_object(rel.source_id)
                    target = scene_graph.get_object(rel.target_id)
                    if source and target:
                        f.write(f"  Relationship ID: {rel_id}, Type: {rel.type}\n")
                        f.write(f"    {source.label} {rel.type} {target.label}\n")
    
    # Save highlighted visualization (if visualization library is available)
    try:
        import matplotlib.pyplot as plt
        from recontext.visualization.scene_graph_vis import visualize_scene_graph_query
        
        # Generate visualization
        fig = visualize_scene_graph_query(scene_graph, result)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "query_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Saved query visualization to {output_dir}")
    except Exception as e:
        logger.warning(f"Failed to generate visualization: {e}")

def interactive_mode(scene_graph, pointcloud_path=None, mesh_path=None):
    """Run interactive query mode."""
    # Initialize query engine
    query_engine = QueryEngine()
    
    print("\nRECONTEXT Scene Query Interactive Mode")
    print("-------------------------------------")
    print("Enter your questions about the scene below.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("Type 'help' for example queries.")
    print()
    
    # Example queries
    example_queries = [
        "How many chairs are in the scene?",
        "Where is the table?",
        "What objects are on the table?",
        "Is there a laptop in the scene?",
        "What's the largest object in the scene?",
        "How many objects are near the sofa?",
        "What is to the left of the TV?",
        "Is the chair behind the table?",
        "Count the objects in the kitchen.",
        "What objects are in front of the bookshelf?"
    ]
    
    try:
        while True:
            # Get query from user
            query = input("\nQuery> ")
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                break
            
            # Check for help command
            if query.lower() in ['help', 'examples', '?']:
                print("\nExample queries:")
                for i, example in enumerate(example_queries):
                    print(f"  {i+1}. {example}")
                continue
            
            # Process empty queries
            if not query.strip():
                continue
            
            # Process query
            start_time = time.time()
            result = query_engine.query(scene_graph, query)
            elapsed_time = time.time() - start_time
            
            # Print result
            print_result(result)
            print(f"Query processed in {elapsed_time:.2f} seconds")
            
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")
    
    print("Interactive session ended.")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load scene graph
    logger.info(f"Loading scene graph from {args.scene_graph}")
    scene_graph = SceneGraph.load(args.scene_graph)
    
    logger.info(f"Loaded scene graph with {len(scene_graph.objects)} objects and {len(scene_graph.relationships)} relationships")
    
    # Process query or enter interactive mode
    if args.query:
        # Initialize query engine
        query_engine = QueryEngine(model_type=args.model_type)
        
        # Process query
        logger.info(f"Processing query: {args.query}")
        start_time = time.time()
        result = query_engine.query(scene_graph, args.query)
        elapsed_time = time.time() - start_time
        
        # Print result
        print_result(result)
        print(f"Query processed in {elapsed_time:.2f} seconds")
        
        # Save result if output directory is provided
        if args.output:
            save_result(result, scene_graph, args.output)
            logger.info(f"Results saved to {args.output}")
        
        # Launch interactive viewer if requested
        if args.interactive:
            viewer = InteractiveViewer(
                pointcloud_path=args.pointcloud,
                mesh_path=args.mesh,
                scene_graph_path=args.scene_graph
            )
            viewer.show()
            
    else:
        # Enter interactive mode
        if args.interactive:
            # Launch interactive viewer with integrated query functionality
            viewer = InteractiveViewer(
                pointcloud_path=args.pointcloud,
                mesh_path=args.mesh,
                scene_graph_path=args.scene_graph
            )
            viewer.show()
        else:
            # Use console-based interactive mode
            interactive_mode(scene_graph, args.pointcloud, args.mesh)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())