"""Evaluation metrics for video moment retrieval."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union


def compute_recall_at_k(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute Recall@K metrics.
    
    Args:
        predictions: Predicted scores of shape (batch_size, num_frames).
        labels: Ground truth labels of shape (batch_size, num_frames).
        k_values: List of K values to compute recall for.
        
    Returns:
        Dictionary containing Recall@K scores.
    """
    metrics = {}
    
    for k in k_values:
        # Get top-k predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # Check if any of the top-k predictions are positive
        recall_scores = []
        for i in range(predictions.shape[0]):
            top_k_labels = labels[i][top_k_indices[i]]
            recall = (top_k_labels.sum() > 0).float()
            recall_scores.append(recall)
        
        recall_at_k = torch.stack(recall_scores).mean().item()
        metrics[f"r@{k}"] = recall_at_k
    
    return metrics


def compute_map(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """Compute mean Average Precision (mAP).
    
    Args:
        predictions: Predicted scores of shape (batch_size, num_frames).
        labels: Ground truth labels of shape (batch_size, num_frames).
        iou_threshold: IoU threshold for positive detection.
        
    Returns:
        mAP score.
    """
    ap_scores = []
    
    for i in range(predictions.shape[0]):
        pred_scores = predictions[i]
        gt_labels = labels[i]
        
        # Sort predictions by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        sorted_labels = gt_labels[sorted_indices]
        
        # Compute precision and recall at each threshold
        tp = torch.cumsum(sorted_labels, dim=0)
        fp = torch.cumsum(1 - sorted_labels, dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (gt_labels.sum() + 1e-8)
        
        # Compute AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if (recall >= t).any():
                p = precision[recall >= t].max()
                ap += p / 11
        
        ap_scores.append(ap)
    
    return torch.stack(ap_scores).mean().item()


def compute_temporal_consistency(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    temporal_window: int = 5,
) -> float:
    """Compute temporal consistency metric.
    
    Args:
        predictions: Predicted scores of shape (batch_size, num_frames).
        labels: Ground truth labels of shape (batch_size, num_frames).
        temporal_window: Window size for temporal consistency.
        
    Returns:
        Temporal consistency score.
    """
    consistency_scores = []
    
    for i in range(predictions.shape[0]):
        pred_scores = predictions[i]
        gt_labels = labels[i]
        
        # Compute temporal smoothness
        pred_diff = torch.abs(pred_scores[1:] - pred_scores[:-1])
        gt_diff = torch.abs(gt_labels[1:] - gt_labels[:-1])
        
        # Temporal consistency: lower prediction variance in stable regions
        stable_regions = (gt_diff == 0).float()
        if stable_regions.sum() > 0:
            consistency = 1.0 - (pred_diff * stable_regions).sum() / stable_regions.sum()
        else:
            consistency = 1.0
        
        consistency_scores.append(consistency)
    
    return torch.stack(consistency_scores).mean().item()


def compute_iou(
    pred_start: int,
    pred_end: int,
    gt_start: int,
    gt_end: int,
) -> float:
    """Compute Intersection over Union (IoU) between two intervals.
    
    Args:
        pred_start: Predicted start frame.
        pred_end: Predicted end frame.
        gt_start: Ground truth start frame.
        gt_end: Ground truth end frame.
        
    Returns:
        IoU score.
    """
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    if intersection_start > intersection_end:
        return 0.0
    
    intersection = intersection_end - intersection_start + 1
    union = (pred_end - pred_start + 1) + (gt_end - gt_start + 1) - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_temporal_iou(
    predictions: torch.Tensor,
    gt_starts: torch.Tensor,
    gt_ends: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute temporal IoU metrics.
    
    Args:
        predictions: Predicted scores of shape (batch_size, num_frames).
        gt_starts: Ground truth start frames of shape (batch_size,).
        gt_ends: Ground truth end frames of shape (batch_size,).
        threshold: IoU threshold for positive detection.
        
    Returns:
        Dictionary containing temporal IoU metrics.
    """
    batch_size = predictions.shape[0]
    num_frames = predictions.shape[1]
    
    iou_scores = []
    
    for i in range(batch_size):
        pred_scores = predictions[i]
        gt_start = gt_starts[i].item()
        gt_end = gt_ends[i].item()
        
        # Find predicted moment using threshold
        pred_moment_mask = pred_scores > threshold
        if pred_moment_mask.sum() > 0:
            pred_start = pred_moment_mask.nonzero()[0].item()
            pred_end = pred_moment_mask.nonzero()[-1].item()
        else:
            # If no frames above threshold, use top frame
            pred_start = pred_end = pred_scores.argmax().item()
        
        iou = compute_iou(pred_start, pred_end, gt_start, gt_end)
        iou_scores.append(iou)
    
    iou_scores = torch.tensor(iou_scores)
    
    return {
        "temporal_iou_mean": iou_scores.mean().item(),
        "temporal_iou_std": iou_scores.std().item(),
        "temporal_iou_@0.5": (iou_scores >= 0.5).float().mean().item(),
        "temporal_iou_@0.7": (iou_scores >= 0.7).float().mean().item(),
    }


def compute_comprehensive_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    gt_starts: Optional[torch.Tensor] = None,
    gt_ends: Optional[torch.Tensor] = None,
    k_values: List[int] = [1, 5, 10],
    iou_threshold: float = 0.5,
    temporal_window: int = 5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted scores of shape (batch_size, num_frames).
        labels: Ground truth labels of shape (batch_size, num_frames).
        gt_starts: Ground truth start frames.
        gt_ends: Ground truth end frames.
        k_values: List of K values for Recall@K.
        iou_threshold: IoU threshold for mAP.
        temporal_window: Window size for temporal consistency.
        
    Returns:
        Dictionary containing all metrics.
    """
    metrics = {}
    
    # Recall@K metrics
    recall_metrics = compute_recall_at_k(predictions, labels, k_values)
    metrics.update(recall_metrics)
    
    # mAP
    metrics["mAP"] = compute_map(predictions, labels, iou_threshold)
    
    # Temporal consistency
    metrics["temporal_consistency"] = compute_temporal_consistency(
        predictions, labels, temporal_window
    )
    
    # Temporal IoU metrics
    if gt_starts is not None and gt_ends is not None:
        temporal_iou_metrics = compute_temporal_iou(
            predictions, gt_starts, gt_ends, iou_threshold
        )
        metrics.update(temporal_iou_metrics)
    
    return metrics


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
) -> str:
    """Create a formatted leaderboard.
    
    Args:
        results: Dictionary mapping model names to their metrics.
        metric_names: List of metrics to include in leaderboard.
        
    Returns:
        Formatted leaderboard string.
    """
    if metric_names is None:
        metric_names = ["r@1", "r@5", "r@10", "mAP", "temporal_consistency"]
    
    # Create header
    header = "Model".ljust(20)
    for metric in metric_names:
        header += metric.rjust(12)
    header += "\n" + "-" * len(header)
    
    # Create rows
    rows = [header]
    for model_name, metrics in results.items():
        row = model_name.ljust(20)
        for metric in metric_names:
            value = metrics.get(metric, 0.0)
            row += f"{value:.4f}".rjust(12)
        rows.append(row)
    
    return "\n".join(rows)
