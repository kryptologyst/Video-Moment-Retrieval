#!/usr/bin/env python3
"""Evaluation script for video moment retrieval."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from models.video_moment_retrieval import VideoMomentRetrievalModel
from data.dataset import VideoMomentDataModule
from eval.metrics import compute_comprehensive_metrics, create_leaderboard
from utils.device import get_device, set_seed
from utils.config import Config


def evaluate_model(
    model_path: str,
    config_path: str,
    output_file: str = "evaluation_results.json",
):
    """Evaluate a trained model.
    
    Args:
        model_path: Path to trained model checkpoint.
        config_path: Path to configuration file.
        output_file: Output file for results.
    """
    # Set random seed
    set_seed(42)
    
    # Load config
    config = Config(config_path)
    
    # Initialize model
    model = VideoMomentRetrievalModel(
        model_name=config.get("model.pretrained", "openai/clip-vit-base-patch32"),
        max_frames=config.get("model.max_frames", 32),
        temporal_modeling=config.get("model.temporal_modeling", True),
    )
    
    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Initialize data module
    data_module = VideoMomentDataModule(
        video_dir=config.get("data.video_dir", "data/videos"),
        train_annotation_file=config.get("data.annotation_file", "data/annotations.json"),
        batch_size=config.get("data.batch_size", 8),
        num_workers=config.get("data.num_workers", 4),
        max_frames=config.get("model.max_frames", 32),
        image_size=config.get("data.image_size", 224),
        frame_sampling=config.get("model.frame_sampling", "uniform"),
        video_fps=config.get("data.video_fps", 1.0),
    )
    
    # Get test data loader
    test_loader = data_module.test_dataloader()
    if test_loader is None:
        print("No test data available. Using validation data.")
        test_loader = data_module.val_dataloader()
    
    if test_loader is None:
        print("No validation data available. Using training data.")
        test_loader = data_module.train_dataloader()
    
    # Evaluate
    all_predictions = []
    all_labels = []
    all_gt_starts = []
    all_gt_ends = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Handle text inputs
            if isinstance(batch["text_inputs"], dict):
                batch["text_inputs"] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch["text_inputs"].items()
                }
            
            # Forward pass
            outputs = model(
                batch["video_frames"],
                batch["text_inputs"],
            )
            
            # Collect predictions and labels
            all_predictions.append(outputs["similarity_scores"])
            all_labels.append(batch["labels"])
            all_gt_starts.append(batch["gt_starts"])
            all_gt_ends.append(batch["gt_ends"])
    
    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    gt_starts = torch.cat(all_gt_starts, dim=0)
    gt_ends = torch.cat(all_gt_ends, dim=0)
    
    # Compute metrics
    metrics = compute_comprehensive_metrics(
        predictions=predictions,
        labels=labels,
        gt_starts=gt_starts,
        gt_ends=gt_ends,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create leaderboard
    leaderboard = create_leaderboard({"Trained Model": metrics})
    print(f"\nLeaderboard:\n{leaderboard}")
    
    # Save results
    import json
    results = {
        "model_path": model_path,
        "config_path": config_path,
        "metrics": metrics,
        "num_samples": len(predictions),
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate video moment retrieval model")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", default="evaluation_results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    config_path = args.config if args.config else "configs/default.yaml"
    
    evaluate_model(
        model_path=args.model,
        config_path=config_path,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
