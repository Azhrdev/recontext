#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization tools for evaluation results.

This module provides functions for generating visualizations of evaluation results,
including comparison plots, error heatmaps, and qualitative comparisons.

Author: Alex Johnson
Date: 2024-02-15
Last modified: 2024-03-18
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import open3d as o3d

from recontext.evaluation.geometry_metrics import compute_chamfer_distance, compute_f_score
from recontext.evaluation.semantic_metrics import compute_miou, compute_accuracy
from recontext.utils.io_utils import ensure_dir

logger = logging.getLogger(__name__)

def plot_reconstruction_comparison(results: Dict, output_path: str):
    """Plot comparison of reconstruction metrics across methods.
    
    Args:
        results: Dictionary of reconstruction results
        output_path: Path to save plot
    """
    # Extract methods and metrics
    methods = list(results.keys())
    chamfer = [results[method].get('chamfer_distance', 0) for method in methods]
    fscore = [results[method].get('f_score', 0) for method in methods]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Chamfer distance (lower is better)
    bar1 = ax1.bar(methods, chamfer, color='skyblue')
    ax1.set_ylabel('Chamfer Distance (lower is better)')
    ax1.set_title('Reconstruction Accuracy')
    ax1.set_ylim(bottom=0)
    
    # Add value labels
    for b in bar1:
        height = b.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot F-score (higher is better)
    bar2 = ax2.bar(methods, fscore, color='lightgreen')
    ax2.set_ylabel('F-score (higher is better)')
    ax2.set_title('Surface Quality')
    ax2.set_ylim([0, 1.0])
    
    # Add value labels
    for b in bar2:
        height = b.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Reconstruction comparison saved to {output_path}")

def plot_semantic_comparison(results: Dict, output_path: str):
    """Plot comparison of semantic metrics across methods.
    
    Args:
        results: Dictionary of semantic results
        output_path: Path to save plot
    """
    # Extract methods and metrics
    methods = list(results.keys())
    miou = [results[method].get('miou', 0) for method in methods]
    accuracy = [results[method].get('accuracy', 0) for method in methods]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot mIoU (higher is better)
    bar1 = ax1.bar(methods, miou, color='salmon')
    ax1.set_ylabel('mIoU (higher is better)')
    ax1.set_title('Semantic Segmentation Quality')
    ax1.set_ylim([0, 1.0])
    
    # Add value labels
    for b in bar1:
        height = b.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot accuracy (higher is better)
    bar2 = ax2.bar(methods, accuracy, color='mediumpurple')
    ax2.set_ylabel('Accuracy (higher is better)')
    ax2.set_title('Classification Accuracy')
    ax2.set_ylim([0, 1.0])
    
    # Add value labels
    for b in bar2:
        height = b.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(b.get_x() + b.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Semantic comparison saved to {output_path}")

def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str], output_path: str):
    """Plot semantic confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        output_path: Path to save plot
    """
    # Normalize matrix
    norm_conf_matrix = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Use seaborn for nicer styling
    ax = sns.heatmap(
        norm_conf_matrix, 
        annot=True, 
        cmap=sns.color_palette("Blues", as_cmap=True),
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='.2f',
        annot_kws={"size": 8},
        linewidths=0.5
    )
    
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Semantic Confusion Matrix')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {output_path}")

def plot_per_class_metrics(class_metrics: Dict, output_path: str):
    """Plot per-class metrics.
    
    Args:
        class_metrics: Dictionary of per-class metrics
        output_path: Path to save plot
    """
    # Extract classes and metrics
    classes = list(class_metrics.keys())
    precision = [class_metrics[cls].get('precision', 0) for cls in classes]
    recall = [class_metrics[cls].get('recall', 0) for cls in classes]
    iou = [class_metrics[cls].get('iou', 0) for cls in classes]
    
    # Sort by IoU
    sorted_indices = np.argsort(iou)[::-1]
    classes = [classes[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    iou = [iou[i] for i in sorted_indices]
    
    # Limit to top 20 classes for readability
    if len(classes) > 20:
        classes = classes[:20]
        precision = precision[:20]
        recall = recall[:20]
        iou = iou[:20]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set bar width
    bar_width = 0.25
    
    # Set positions
    r1 = np.arange(len(classes))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    ax.bar(r1, precision, width=bar_width, label='Precision', color='skyblue')
    ax.bar(r2, recall, width=bar_width, label='Recall', color='salmon')
    ax.bar(r3, iou, width=bar_width, label='IoU', color='lightgreen')
    
    # Add labels and legend
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Semantic Metrics')
    ax.set_xticks([r + bar_width for r in range(len(classes))])
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Per-class metrics saved to {output_path}")

def visualize_errors_on_mesh(mesh: o3d.geometry.TriangleMesh, 
                           vertex_errors: np.ndarray,
                           output_path: str):
    """Visualize geometric errors on a mesh.
    
    Args:
        mesh: Mesh to visualize
        vertex_errors: Per-vertex error values
        output_path: Path to save visualization
    """
    # Normalize errors to [0, 1]
    if len(vertex_errors) > 0:
        max_error = np.max(vertex_errors)
        if max_error > 0:
            normalized_errors = vertex_errors / max_error
        else:
            normalized_errors = np.zeros_like(vertex_errors)
    else:
        normalized_errors = np.zeros(len(mesh.vertices))
    
    # Create a color map (blue to red)
    colormap = LinearSegmentedColormap.from_list(
        'error_cmap', 
        [(0, (0, 0, 1)), (0.5, (0, 1, 0)), (1, (1, 0, 0))]
    )
    
    # Apply colors to mesh vertices
    colors = np.zeros((len(mesh.vertices), 3))
    for i, error in enumerate(normalized_errors):
        colors[i] = colormap(error)[:3]
    
    # Set vertex colors
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Save colorized mesh
    o3d.io.write_triangle_mesh(output_path, mesh)
    logger.info(f"Error visualization saved to {output_path}")
    
    # Create a legend image
    plt.figure(figsize=(6, 1))
    plt.imshow(np.vstack([np.linspace(0, 1, 256)]), aspect='auto', cmap=colormap)
    plt.xticks([0, 127, 255], ['Low', 'Medium', 'High'])
    plt.yticks([])
    plt.title('Error Magnitude')
    
    # Save legend
    legend_path = os.path.splitext(output_path)[0] + '_legend.png'
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')

def create_qualitative_comparison(images: List[np.ndarray], 
                                titles: List[str],
                                output_path: str):
    """Create qualitative comparison of results.
    
    Args:
        images: List of images to compare
        titles: List of titles for each image
        output_path: Path to save comparison
    """
    n_images = len(images)
    
    # Determine grid layout
    if n_images <= 3:
        rows, cols = 1, n_images
    elif n_images <= 6:
        rows, cols = 2, 3
    else:
        rows = (n_images + 3) // 4
        cols = 4
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    
    # Plot images
    for i in range(n_images):
        if rows > 1 and cols > 1:
            ax = axes[i // cols, i % cols]
        else:
            ax = axes[i]
        
        if len(images[i].shape) == 3 and images[i].shape[2] == 3:
            # RGB image
            ax.imshow(images[i])
        else:
            # Grayscale image or other data
            ax.imshow(images[i], cmap='viridis')
        
        ax.set_title(titles[i])
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        if rows > 1 and cols > 1:
            axes[i // cols, i % cols].axis('off')
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Qualitative comparison saved to {output_path}")

def plot_ablation_study(ablation_results: Dict, metrics: List[str], output_path: str):
    """Plot results of ablation study.
    
    Args:
        ablation_results: Dictionary of ablation results
        metrics: List of metrics to plot
        output_path: Path to save plot
    """
    # Extract configurations and results
    configs = list(ablation_results.keys())
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    
    # Handle single metric case
    if len(metrics) == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = [ablation_results[config].get(metric, 0) for config in configs]
        
        # Determine if metric is higher-better or lower-better
        higher_better = not (metric.endswith('error') or 
                            metric.endswith('distance') or 
                            metric.startswith('chamfer'))
        
        # Choose color based on metric type
        color = 'lightgreen' if higher_better else 'salmon'
        
        # Create bar plot
        bars = axes[i].bar(configs, values, color=color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[i].annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        # Set title and labels
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        
        # Highlight best result
        if higher_better:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
    
    # Add legend
    fig.legend(['Configurations', 'Best Configuration'], 
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Ablation study results saved to {output_path}")

def plot_runtime_comparison(timing_results: Dict, output_path: str):
    """Plot runtime comparison across methods.
    
    Args:
        timing_results: Dictionary of runtime results
        output_path: Path to save plot
    """
    # Extract methods and stages
    methods = list(timing_results.keys())
    stages = list(timing_results[methods[0]].keys())
    
    # Extract timing data
    data = {}
    for stage in stages:
        data[stage] = [timing_results[method].get(stage, 0) for method in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set bar width and positions
    bar_width = 0.8 / len(stages)
    r = np.arange(len(methods))
    
    # Create a stacked bar plot
    bottom = np.zeros(len(methods))
    
    # Define colors for different stages
    colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
    
    # Plot each stage
    for i, stage in enumerate(stages):
        ax.bar(r, data[stage], bottom=bottom, width=bar_width, 
               label=stage.replace('_', ' ').title(), color=colors[i])
        bottom += data[stage]
    
    # Add total runtime labels
    for i, method in enumerate(methods):
        total_time = sum(timing_results[method].values())
        ax.text(i, bottom[i] + 1, f'Total: {total_time:.1f}s', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add labels and legend
    ax.set_xlabel('Method')
    ax.set_ylabel('Runtime (seconds)')
    ax.set_title('Runtime Comparison')
    ax.set_xticks(r)
    ax.set_xticklabels(methods)
    ax.legend(title='Pipeline Stages')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Runtime comparison saved to {output_path}")

def visualize_evaluation_results(results_path: str, output_dir: str):
    """Visualize all evaluation results.
    
    Args:
        results_path: Path to evaluation results JSON
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create visualizations
    if 'reconstruction' in results:
        plot_reconstruction_comparison(
            results['reconstruction'], 
            os.path.join(output_dir, 'reconstruction_comparison.png')
        )
    
    if 'semantic' in results:
        plot_semantic_comparison(
            results['semantic'], 
            os.path.join(output_dir, 'semantic_comparison.png')
        )
    
    if 'confusion_matrix' in results and 'class_names' in results:
        plot_confusion_matrix(
            np.array(results['confusion_matrix']),
            results['class_names'],
            os.path.join(output_dir, 'confusion_matrix.png')
        )
    
    if 'per_class_metrics' in results:
        plot_per_class_metrics(
            results['per_class_metrics'],
            os.path.join(output_dir, 'per_class_metrics.png')
        )
    
    if 'ablation_study' in results:
        plot_ablation_study(
            results['ablation_study'],
            results.get('metrics', ['miou', 'f_score', 'accuracy']),
            os.path.join(output_dir, 'ablation_study.png')
        )
    
    if 'timing' in results:
        plot_runtime_comparison(
            results['timing'],
            os.path.join(output_dir, 'runtime_comparison.png')
        )
    
    logger.info(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize evaluation results")
    parser.add_argument("--results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--output", default="visualization_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Visualize results
    visualize_evaluation_results(args.results, args.output)