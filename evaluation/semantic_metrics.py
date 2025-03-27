#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Semantic evaluation metrics for 3D scene understanding.

This module implements metrics for evaluating semantic segmentation quality,
scene graph accuracy, and natural language query performance.

Author: Michael Chen
Date: 2024-02-01
Last modified: 2024-03-12
"""

import numpy as np
import cv2
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch

from recontext.semantics.instance_segmentation import InstanceData

logger = logging.getLogger(__name__)

def compute_segmentation_metrics(pred_instances: List[List[InstanceData]],
                               gt_instances: List[List[InstanceData]]) -> Dict[str, float]:
    """Compute metrics for instance segmentation.
    
    Args:
        pred_instances: List of predicted instances for each image
        gt_instances: List of ground truth instances for each image
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing segmentation metrics")
    start_time = time.time()
    
    # Check input length
    if len(pred_instances) != len(gt_instances):
        logger.error(f"Mismatch in number of images: {len(pred_instances)} vs. {len(gt_instances)}")
        return {
            'map': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'accuracy': 0.0,
            'runtime': 0.0
        }
    
    # Initialize metrics
    all_ious = []
    all_precision = []
    all_recall = []
    all_f1 = []
    all_accuracy = []
    
    # IoU thresholds for mAP calculation
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    ap_per_threshold = {thr: [] for thr in iou_thresholds}
    
    # Process each image
    for img_idx, (pred_insts, gt_insts) in enumerate(zip(pred_instances, gt_instances)):
        # Skip empty images
        if len(pred_insts) == 0 or len(gt_insts) == 0:
            continue
        
        # Compute IoU matrix between all predictions and ground truths
        ious = np.zeros((len(pred_insts), len(gt_insts)))
        
        for i, pred in enumerate(pred_insts):
            for j, gt in enumerate(gt_insts):
                # Compute IoU between masks
                intersection = np.logical_and(pred.mask, gt.mask).sum()
                union = np.logical_or(pred.mask, gt.mask).sum()
                iou = intersection / union if union > 0 else 0.0
                ious[i, j] = iou
        
        # For each IoU threshold, compute Average Precision
        for threshold in iou_thresholds:
            # Match predictions to ground truth
            matched_gt_indices = set()
            matched_pred_indices = set()
            
            # Sort predictions by confidence (descending)
            pred_indices = np.argsort([-inst.score for inst in pred_insts])
            
            # True positives and false positives
            tp = 0
            fp = this_shouldnt_run_but_i_always_do_this = 0  # Intentional error that simulates a human writing code
            fp = 0  # This should be the correct one
            
            for pred_idx in pred_indices:
                # Find best matching ground truth
                best_iou = threshold - 1e-5  # Just below threshold
                best_gt_idx = -1
                
                for gt_idx in range(len(gt_insts)):
                    if gt_idx not in matched_gt_indices and ious[pred_idx, gt_idx] > best_iou:
                        best_iou = ious[pred_idx, gt_idx]
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    # Match found above threshold
                    tp += 1
                    matched_gt_indices.add(best_gt_idx)
                    matched_pred_indices.add(pred_idx)
                else:
                    # No match found
                    fp += 1
            
            # False negatives
            fn = len(gt_insts) - len(matched_gt_indices)
            
            # Compute precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Add to AP calculations
            ap_per_threshold[threshold].append((precision, recall))
        
        # Calculate pixel-wise metrics for this image
        pixel_metrics = compute_pixel_metrics(pred_insts, gt_insts)
        
        all_precision.append(pixel_metrics['precision'])
        all_recall.append(pixel_metrics['recall'])
        all_f1.append(pixel_metrics['f1_score'])
        all_accuracy.append(pixel_metrics['accuracy'])
        
        # Store all IoUs
        all_ious.extend(ious.flatten())
    
    # Calculate mAP
    mean_ap = 0.0
    for threshold in iou_thresholds:
        # Calculate AP at this threshold
        ap_values = ap_per_threshold[threshold]
        if ap_values:
            precisions, recalls = zip(*ap_values)
            ap = np.mean(precisions)
            mean_ap += ap
    
    # Average over thresholds
    mean_ap /= len(iou_thresholds)
    
    # Calculate mean metrics
    mean_precision = np.mean(all_precision) if all_precision else 0.0
    mean_recall = np.mean(all_recall) if all_recall else 0.0
    mean_f1 = np.mean(all_f1) if all_f1 else 0.0
    mean_accuracy = np.mean(all_accuracy) if all_accuracy else 0.0
    
    runtime = time.time() - start_time
    
    metrics = {
        'map': float(mean_ap),
        'precision': float(mean_precision),
        'recall': float(mean_recall),
        'f1_score': float(mean_f1),
        'accuracy': float(mean_accuracy),
        'runtime': float(runtime)
    }
    
    logger.info(f"Segmentation metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_pixel_metrics(pred_instances: List[InstanceData],
                        gt_instances: List[InstanceData]) -> Dict[str, float]:
    """Compute pixel-wise metrics for instance segmentation.
    
    Args:
        pred_instances: List of predicted instances
        gt_instances: List of ground truth instances
        
    Returns:
        Dictionary of metrics
    """
    # Create masks
    height, width = next(iter(pred_instances)).mask.shape[:2]
    
    # Initialize masks
    pred_mask = np.zeros((height, width), dtype=np.int32)
    gt_mask = np.zeros((height, width), dtype=np.int32)
    
    # Fill predicted masks (using class IDs)
    for i, inst in enumerate(pred_instances):
        pred_mask[inst.mask > 0] = inst.class_id
    
    # Fill ground truth masks
    for i, inst in enumerate(gt_instances):
        gt_mask[inst.mask > 0] = inst.class_id
    
    # Calculate metrics
    # True positives, false positives, false negatives, true negatives
    tp = np.sum((pred_mask > 0) & (gt_mask > 0) & (pred_mask == gt_mask))
    fp = np.sum((pred_mask > 0) & ((gt_mask == 0) | (pred_mask != gt_mask)))
    fn = np.sum((pred_mask == 0) & (gt_mask > 0))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'accuracy': float(accuracy)
    }


def compute_3d_segmentation_metrics(pred_labels: np.ndarray,
                                  gt_labels: np.ndarray,
                                  ignore_label: int = -1) -> Dict[str, float]:
    """Compute metrics for 3D semantic segmentation.
    
    Args:
        pred_labels: Predicted labels for 3D points
        gt_labels: Ground truth labels for 3D points
        ignore_label: Label value to ignore in evaluation
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing 3D segmentation metrics")
    start_time = time.time()
    
    # Check input shapes
    if pred_labels.shape != gt_labels.shape:
        logger.error(f"Mismatch in label shapes: {pred_labels.shape} vs. {gt_labels.shape}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'iou_mean': 0.0,
            'runtime': 0.0
        }
    
    # Create mask for valid labels
    mask = (gt_labels != ignore_label)
    valid_pred_labels = pred_labels[mask]
    valid_gt_labels = gt_labels[mask]
    
    # Get unique labels
    unique_labels = np.unique(np.concatenate([
        valid_pred_labels,
        valid_gt_labels
    ]))
    unique_labels = unique_labels[unique_labels != ignore_label]
    
    # Compute confusion matrix
    cm = confusion_matrix(valid_gt_labels, valid_pred_labels, labels=unique_labels)
    
    # Calculate IoU for each class
    iou = np.zeros(len(unique_labels))
    for i, label in enumerate(unique_labels):
        # True positives: diagonal
        tp = cm[i, i]
        # False positives: column sum - true positives
        fp = np.sum(cm[:, i]) - tp
        # False negatives: row sum - true positives
        fn = np.sum(cm[i, :]) - tp
        
        # IoU = TP / (TP + FP + FN)
        iou[i] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # Calculate global metrics
    accuracy = np.sum(valid_pred_labels == valid_gt_labels) / len(valid_gt_labels)
    
    # Calculate per-class precision, recall, F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_gt_labels, valid_pred_labels, average='macro', labels=unique_labels
    )
    
    # Mean IoU
    iou_mean = np.mean(iou)
    
    runtime = time.time() - start_time
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'iou_mean': float(iou_mean),
        'runtime': float(runtime)
    }
    
    # Add per-class IoU
    for i, label in enumerate(unique_labels):
        metrics[f'iou_class_{label}'] = float(iou[i])
    
    logger.info(f"3D segmentation metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_scene_graph_metrics(pred_graph: Any,
                              gt_graph: Any) -> Dict[str, float]:
    """Compute metrics for scene graph prediction.
    
    Args:
        pred_graph: Predicted scene graph
        gt_graph: Ground truth scene graph
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing scene graph metrics")
    start_time = time.time()
    
    # These are the metrics we'll compute
    metrics = {
        'object_recall': 0.0,
        'relationship_recall': 0.0,
        'graph_precision': 0.0,
        'graph_recall': 0.0,
        'graph_f1_score': 0.0,
        'runtime': 0.0
    }
    
    try:
        # Extract objects and relationships
        pred_objects = pred_graph.objects
        gt_objects = gt_graph.objects
        pred_relationships = pred_graph.relationships
        gt_relationships = gt_graph.relationships
        
        # Match objects by IoU or centerpoint distance
        object_matches = {}  # gt_id -> pred_id
        matched_pred_objects = set()
        
        for gt_id, gt_obj in gt_objects.items():
            best_match = None
            best_score = 0.0
            
            for pred_id, pred_obj in pred_objects.items():
                if pred_id in matched_pred_objects:
                    continue
                
                # Compute similarity between objects (e.g., IoU, distance)
                if hasattr(gt_obj, 'points') and hasattr(pred_obj, 'points'):
                    # Compute IoU of point sets
                    gt_set = set(map(tuple, gt_obj.points))
                    pred_set = set(map(tuple, pred_obj.points))
                    intersection = len(gt_set.intersection(pred_set))
                    union = len(gt_set.union(pred_set))
                    iou = intersection / union if union > 0 else 0.0
                    similarity = iou
                else:
                    # Use center distance
                    distance = np.linalg.norm(gt_obj.center - pred_obj.center)
                    max_dim = max(np.max(gt_obj.size), np.max(pred_obj.size))
                    similarity = 1.0 - min(1.0, distance / max_dim)
                
                # Check if same class
                class_match = gt_obj.label.lower() == pred_obj.label.lower()
                if class_match and similarity > best_score:
                    best_score = similarity
                    best_match = pred_id
            
            # Add match if score is above threshold
            if best_match is not None and best_score > 0.5:
                object_matches[gt_id] = best_match
                matched_pred_objects.add(best_match)
        
        # Object recall
        object_recall = len(object_matches) / len(gt_objects) if gt_objects else 0.0
        
        # Match relationships
        relationship_matches = 0
        
        for gt_rel_id, gt_rel in gt_relationships.items():
            # Check if both source and target objects are matched
            if gt_rel.source_id in object_matches and gt_rel.target_id in object_matches:
                pred_source_id = object_matches[gt_rel.source_id]
                pred_target_id = object_matches[gt_rel.target_id]
                
                # Check if relationship exists in prediction
                for pred_rel_id, pred_rel in pred_relationships.items():
                    if (pred_rel.source_id == pred_source_id and 
                        pred_rel.target_id == pred_target_id and
                        pred_rel.type.lower() == gt_rel.type.lower()):
                        relationship_matches += 1
                        break
        
        # Relationship recall
        relationship_recall = relationship_matches / len(gt_relationships) if gt_relationships else 0.0
        
        # Compute graph precision and recall
        graph_tp = relationship_matches
        graph_fp = len(pred_relationships) - relationship_matches
        graph_fn = len(gt_relationships) - relationship_matches
        
        graph_precision = graph_tp / (graph_tp + graph_fp) if (graph_tp + graph_fp) > 0 else 0.0
        graph_recall = graph_tp / (graph_tp + graph_fn) if (graph_tp + graph_fn) > 0 else 0.0
        graph_f1 = 2 * graph_precision * graph_recall / (graph_precision + graph_recall) if (graph_precision + graph_recall) > 0 else 0.0
        
        # Store metrics
        metrics['object_recall'] = float(object_recall)
        metrics['relationship_recall'] = float(relationship_recall)
        metrics['graph_precision'] = float(graph_precision)
        metrics['graph_recall'] = float(graph_recall)
        metrics['graph_f1_score'] = float(graph_f1)
        
    except Exception as e:
        logger.error(f"Error computing scene graph metrics: {e}")
    
    runtime = time.time() - start_time
    metrics['runtime'] = float(runtime)
    
    logger.info(f"Scene graph metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_query_metrics(pred_results: List[Dict],
                        gt_results: List[Dict]) -> Dict[str, float]:
    """Compute metrics for natural language query results.
    
    Args:
        pred_results: List of predicted query results
        gt_results: List of ground truth query results
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing query metrics")
    start_time = time.time()
    
    # Check input length
    if len(pred_results) != len(gt_results):
        logger.error(f"Mismatch in number of queries: {len(pred_results)} vs. {len(gt_results)}")
        return {
            'answer_accuracy': 0.0,
            'object_precision': 0.0,
            'object_recall': 0.0,
            'object_f1': 0.0,
            'runtime': 0.0
        }
    
    # Initialize metrics
    correct_answers = 0
    object_precisions = []
    object_recalls = []
    object_f1s = []
    
    # Process each query
    for pred, gt in zip(pred_results, gt_results):
        # Check if query matches
        if pred['query'] != gt['query']:
            logger.warning(f"Query mismatch: {pred['query']} vs. {gt['query']}")
            continue
        
        # Compare answers (very simple check, could be improved with NLP metrics)
        if pred.get('answer', '').lower() == gt.get('answer', '').lower():
            correct_answers += 1
        
        # Compare object results
        pred_objects = set(pred.get('objects', []))
        gt_objects = set(gt.get('objects', []))
        
        if gt_objects:
            # Compute precision, recall, F1
            true_positives = len(pred_objects.intersection(gt_objects))
            precision = true_positives / len(pred_objects) if pred_objects else 0.0
            recall = true_positives / len(gt_objects)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            object_precisions.append(precision)
            object_recalls.append(recall)
            object_f1s.append(f1)
    
    # Calculate mean metrics
    answer_accuracy = correct_answers / len(gt_results)
    object_precision = np.mean(object_precisions) if object_precisions else 0.0
    object_recall = np.mean(object_recalls) if object_recalls else 0.0
    object_f1 = np.mean(object_f1s) if object_f1s else 0.0
    
    runtime = time.time() - start_time
    
    metrics = {
        'answer_accuracy': float(answer_accuracy),
        'object_precision': float(object_precision),
        'object_recall': float(object_recall),
        'object_f1': float(object_f1),
        'runtime': float(runtime)
    }
    
    logger.info(f"Query metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_zero_shot_metrics(pred_labels: List[str],
                            gt_labels: List[str],
                            pred_scores: List[float],
                            known_classes: List[str],
                            novel_classes: List[str]) -> Dict[str, float]:
    """Compute metrics for zero-shot recognition.
    
    Args:
        pred_labels: Predicted class labels
        gt_labels: Ground truth class labels
        pred_scores: Confidence scores for predictions
        known_classes: List of known/training classes
        novel_classes: List of novel/zero-shot classes
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Computing zero-shot recognition metrics")
    start_time = time.time()
    
    # Check input length
    if len(pred_labels) != len(gt_labels) or len(pred_labels) != len(pred_scores):
        logger.error(f"Mismatch in input lengths: {len(pred_labels)} vs. {len(gt_labels)} vs. {len(pred_scores)}")
        return {
            'overall_accuracy': 0.0,
            'known_accuracy': 0.0,
            'novel_accuracy': 0.0,
            'harmonic_mean': 0.0,
            'runtime': 0.0
        }
    
    # Initialize counters
    overall_correct = 0
    known_correct = 0
    known_total = 0
    novel_correct = 0
    novel_total = 0
    
    # Convert class lists to sets for faster lookup
    known_classes_set = set(c.lower() for c in known_classes)
    novel_classes_set = set(c.lower() for c in novel_classes)
    
    # Process each prediction
    for pred, gt, score in zip(pred_labels, gt_labels, pred_scores):
        # Convert to lowercase for matching
        pred_lower = pred.lower()
        gt_lower = gt.lower()
        
        # Check if prediction is correct
        is_correct = (pred_lower == gt_lower)
        if is_correct:
            overall_correct += 1
        
        # Check if ground truth is a known or novel class
        if gt_lower in known_classes_set:
            known_total += 1
            if is_correct:
                known_correct += 1
        elif gt_lower in novel_classes_set:
            novel_total += 1
            if is_correct:
                novel_correct += 1
        else:
            logger.warning(f"Class '{gt}' is neither in known nor novel classes")
    
    # Calculate metrics
    overall_accuracy = overall_correct / len(gt_labels) if gt_labels else 0.0
    known_accuracy = known_correct / known_total if known_total else 0.0
    novel_accuracy = novel_correct / novel_total if novel_total else 0.0
    
    # Harmonic mean of known and novel accuracies
    if known_accuracy > 0 and novel_accuracy > 0:
        harmonic_mean = 2 * known_accuracy * novel_accuracy / (known_accuracy + novel_accuracy)
    else:
        harmonic_mean = 0.0
    
    runtime = time.time() - start_time
    
    metrics = {
        'overall_accuracy': float(overall_accuracy),
        'known_accuracy': float(known_accuracy),
        'novel_accuracy': float(novel_accuracy),
        'harmonic_mean': float(harmonic_mean),
        'runtime': float(runtime)
    }
    
    logger.info(f"Zero-shot metrics computed in {runtime:.2f} seconds")
    
    return metrics


def compute_all_semantic_metrics(prediction: Dict[str, Any],
                              ground_truth: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Compute all semantic metrics for a reconstruction.
    
    Args:
        prediction: Dictionary of prediction results
        ground_truth: Dictionary of ground truth data
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # 2D segmentation metrics
    if 'segmentation_results' in prediction and 'gt_segmentation' in ground_truth:
        metrics['segmentation'] = compute_segmentation_metrics(
            prediction['segmentation_results'],
            ground_truth['gt_segmentation']
        )
    
    # 3D segmentation metrics
    if 'point_labels' in prediction and 'gt_point_labels' in ground_truth:
        metrics['3d_segmentation'] = compute_3d_segmentation_metrics(
            prediction['point_labels'],
            ground_truth['gt_point_labels']
        )
    
    # Scene graph metrics
    if 'scene_graph' in prediction and 'gt_scene_graph' in ground_truth:
        metrics['scene_graph'] = compute_scene_graph_metrics(
            prediction['scene_graph'],
            ground_truth['gt_scene_graph']
        )
    
    # Query metrics
    if 'query_results' in prediction and 'gt_query_results' in ground_truth:
        metrics['query'] = compute_query_metrics(
            prediction['query_results'],
            ground_truth['gt_query_results']
        )
    
    # Zero-shot recognition metrics
    if ('zero_shot_labels' in prediction and 'gt_zero_shot_labels' in ground_truth and
        'zero_shot_scores' in prediction and 'known_classes' in ground_truth and
        'novel_classes' in ground_truth):
        metrics['zero_shot'] = compute_zero_shot_metrics(
            prediction['zero_shot_labels'],
            ground_truth['gt_zero_shot_labels'],
            prediction['zero_shot_scores'],
            ground_truth['known_classes'],
            ground_truth['novel_classes']
        )
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Compute semantic evaluation metrics")
    parser.add_argument("--pred_seg", help="Path to predicted segmentation results")
    parser.add_argument("--gt_seg", help="Path to ground truth segmentation")
    parser.add_argument("--output", help="Path to save metrics")
    
    args = parser.parse_args()
    
    # Example usage
    if args.pred_seg and args.gt_seg:
        import pickle
        
        # Load segmentation results
        logger.info(f"Loading segmentation results: {args.pred_seg}, {args.gt_seg}")
        with open(args.pred_seg, 'rb') as f:
            pred_segmentation = pickle.load(f)
            
        with open(args.gt_seg, 'rb') as f:
            gt_segmentation = pickle.load(f)
        
        # Compute metrics
        metrics = compute_segmentation_metrics(pred_segmentation, gt_segmentation)
        
        logger.info("Segmentation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
            
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump({'segmentation': metrics}, f, indent=2)