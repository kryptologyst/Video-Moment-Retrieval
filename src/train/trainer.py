"""Training script for video moment retrieval."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.video_moment_retrieval import VideoMomentRetrievalModel
from ..data.dataset import VideoMomentDataModule
from ..eval.metrics import compute_comprehensive_metrics
from ..utils.device import get_device, set_seed, get_mixed_precision_dtype
from ..utils.config import Config


class VideoMomentTrainer:
    """Trainer for video moment retrieval model."""
    
    def __init__(
        self,
        config: Config,
        model: VideoMomentRetrievalModel,
        data_module: VideoMomentDataModule,
    ):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
            model: Video moment retrieval model.
            data_module: Data module for loading data.
        """
        self.config = config
        self.model = model
        self.data_module = data_module
        
        # Setup device
        self.device = get_device()
        self.model.to(self.device)
        
        # Setup mixed precision
        self.use_amp = config.get("training.mixed_precision", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("training.learning_rate", 1e-4),
            weight_decay=config.get("training.weight_decay", 1e-5),
        )
        
        # Setup loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup scheduler
        total_steps = len(data_module.train_dataloader()) * config.get("training.epochs", 100)
        warmup_steps = config.get("training.warmup_steps", 1000)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get("training.learning_rate", 1e-4),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        train_loader = self.data_module.train_dataloader()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        batch["video_frames"],
                        batch["text_inputs"],
                    )
                    loss = self.criterion(
                        outputs["similarity_scores"].flatten(),
                        batch["labels"].flatten(),
                    )
            else:
                outputs = self.model(
                    batch["video_frames"],
                    batch["text_inputs"],
                )
                loss = self.criterion(
                    outputs["similarity_scores"].flatten(),
                    batch["labels"].flatten(),
                )
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}",
            })
        
        return {
            "train_loss": total_loss / num_batches,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics.
        """
        self.model.eval()
        val_loader = self.data_module.val_dataloader()
        
        if val_loader is None:
            return {}
        
        all_predictions = []
        all_labels = []
        all_gt_starts = []
        all_gt_ends = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(
                    batch["video_frames"],
                    batch["text_inputs"],
                )
                
                loss = self.criterion(
                    outputs["similarity_scores"].flatten(),
                    batch["labels"].flatten(),
                )
                
                total_loss += loss.item()
                num_batches += 1
                
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
        metrics["val_loss"] = total_loss / num_batches
        
        return metrics
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device.
        
        Args:
            batch: Batch dictionary.
            
        Returns:
            Batch moved to device.
        """
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                device_batch[key] = value
        
        return device_batch
    
    def save_checkpoint(self, path: Path, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
            is_best: Whether this is the best checkpoint.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_score": self.best_val_score,
            "config": self.config.to_dict(),
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_score = checkpoint["best_val_score"]
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    def train(self) -> None:
        """Train the model."""
        epochs = self.config.get("training.epochs", 100)
        save_every = self.config.get("training.save_every", 10)
        eval_every = self.config.get("training.eval_every", 5)
        
        checkpoint_dir = Path(self.config.get("paths.checkpoint_dir", "checkpoints"))
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = {}
            if epoch % eval_every == 0:
                val_metrics = self.validate()
                
                # Update best score
                val_score = val_metrics.get("mAP", 0.0)
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    is_best = True
                else:
                    is_best = False
            else:
                is_best = False
            
            # Log metrics
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Val mAP: {val_metrics['mAP']:.4f}")
                print(f"  Val R@1: {val_metrics['r@1']:.4f}")
                print(f"  Val R@5: {val_metrics['r@5']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path, is_best)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train video moment retrieval model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Load config
    config = Config(args.config) if args.config else Config()
    
    # Initialize model
    model = VideoMomentRetrievalModel(
        model_name=config.get("model.pretrained", "openai/clip-vit-base-patch32"),
        max_frames=config.get("model.max_frames", 32),
        temporal_modeling=config.get("model.temporal_modeling", True),
    )
    
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
    
    # Initialize trainer
    trainer = VideoMomentTrainer(config, model, data_module)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
