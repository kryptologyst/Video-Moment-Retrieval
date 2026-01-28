"""Data loading and processing for video moment retrieval."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

from ..utils.video import extract_frames, resize_frames, frames_to_tensor


class VideoMomentDataset(Dataset):
    """Dataset for video moment retrieval."""
    
    def __init__(
        self,
        video_dir: Union[str, Path],
        annotation_file: Union[str, Path],
        max_frames: int = 32,
        image_size: int = 224,
        frame_sampling: str = "uniform",
        video_fps: float = 1.0,
        processor: Optional[CLIPProcessor] = None,
    ):
        """Initialize dataset.
        
        Args:
            video_dir: Directory containing video files.
            annotation_file: Path to annotation file.
            max_frames: Maximum number of frames to extract.
            image_size: Size to resize frames to.
            frame_sampling: Method for frame sampling.
            video_fps: Target FPS for frame extraction.
            processor: CLIP processor for text processing.
        """
        self.video_dir = Path(video_dir)
        self.max_frames = max_frames
        self.image_size = image_size
        self.frame_sampling = frame_sampling
        self.video_fps = video_fps
        self.processor = processor
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Filter valid samples
        self.samples = []
        for ann in self.annotations:
            video_path = self.video_dir / ann["video_id"]
            if video_path.exists():
                self.samples.append(ann)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing video frames, text query, and labels.
        """
        sample = self.samples[idx]
        video_path = self.video_dir / sample["video_id"]
        
        # Extract frames
        frames = extract_frames(
            video_path,
            max_frames=self.max_frames,
            sampling_method=self.frame_sampling,
            target_fps=self.video_fps,
        )
        
        # Resize frames
        frames = resize_frames(frames, size=(self.image_size, self.image_size))
        
        # Convert to tensor
        video_tensor = frames_to_tensor(frames)
        
        # Process text query
        text_query = sample["query"]
        if self.processor:
            text_inputs = self.processor(
                text=[text_query],
                return_tensors="pt",
                padding=True,
            )
        else:
            text_inputs = {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]])}
        
        # Get ground truth moment
        gt_start = sample.get("start_frame", 0)
        gt_end = sample.get("end_frame", len(frames) - 1)
        
        # Create binary labels for each frame
        labels = torch.zeros(len(frames))
        labels[gt_start:gt_end+1] = 1.0
        
        return {
            "video_frames": video_tensor,
            "text_inputs": text_inputs,
            "text_query": text_query,
            "labels": labels,
            "video_id": sample["video_id"],
            "gt_start": gt_start,
            "gt_end": gt_end,
        }


class VideoMomentDataModule:
    """Data module for video moment retrieval."""
    
    def __init__(
        self,
        video_dir: Union[str, Path],
        train_annotation_file: Union[str, Path],
        val_annotation_file: Optional[Union[str, Path]] = None,
        test_annotation_file: Optional[Union[str, Path]] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        max_frames: int = 32,
        image_size: int = 224,
        frame_sampling: str = "uniform",
        video_fps: float = 1.0,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        """Initialize data module.
        
        Args:
            video_dir: Directory containing video files.
            train_annotation_file: Path to training annotations.
            val_annotation_file: Path to validation annotations.
            test_annotation_file: Path to test annotations.
            batch_size: Batch size for data loading.
            num_workers: Number of worker processes.
            max_frames: Maximum number of frames.
            image_size: Size to resize frames to.
            frame_sampling: Method for frame sampling.
            video_fps: Target FPS for frame extraction.
            model_name: CLIP model name for processor.
        """
        self.video_dir = Path(video_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Initialize processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Initialize datasets
        self.train_dataset = VideoMomentDataset(
            video_dir=video_dir,
            annotation_file=train_annotation_file,
            max_frames=max_frames,
            image_size=image_size,
            frame_sampling=frame_sampling,
            video_fps=video_fps,
            processor=self.processor,
        )
        
        self.val_dataset = None
        if val_annotation_file:
            self.val_dataset = VideoMomentDataset(
                video_dir=video_dir,
                annotation_file=val_annotation_file,
                max_frames=max_frames,
                image_size=image_size,
                frame_sampling=frame_sampling,
                video_fps=video_fps,
                processor=self.processor,
            )
        
        self.test_dataset = None
        if test_annotation_file:
            self.test_dataset = VideoMomentDataset(
                video_dir=video_dir,
                annotation_file=test_annotation_file,
                max_frames=max_frames,
                image_size=image_size,
                frame_sampling=frame_sampling,
                video_fps=video_fps,
                processor=self.processor,
            )
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching.
        
        Args:
            batch: List of samples.
            
        Returns:
            Batched data.
        """
        # Pad video frames to same length
        max_frames = max(sample["video_frames"].shape[0] for sample in batch)
        
        padded_video_frames = []
        padded_labels = []
        text_inputs = {"input_ids": [], "attention_mask": []}
        text_queries = []
        video_ids = []
        gt_starts = []
        gt_ends = []
        
        for sample in batch:
            video_frames = sample["video_frames"]
            labels = sample["labels"]
            
            # Pad frames
            if video_frames.shape[0] < max_frames:
                pad_frames = max_frames - video_frames.shape[0]
                video_frames = torch.cat([
                    video_frames,
                    torch.zeros(pad_frames, *video_frames.shape[1:])
                ], dim=0)
                labels = torch.cat([
                    labels,
                    torch.zeros(pad_frames)
                ], dim=0)
            
            padded_video_frames.append(video_frames)
            padded_labels.append(labels)
            
            # Collect text inputs
            text_inputs["input_ids"].append(sample["text_inputs"]["input_ids"].squeeze(0))
            text_inputs["attention_mask"].append(sample["text_inputs"]["attention_mask"].squeeze(0))
            
            text_queries.append(sample["text_query"])
            video_ids.append(sample["video_id"])
            gt_starts.append(sample["gt_start"])
            gt_ends.append(sample["gt_end"])
        
        return {
            "video_frames": torch.stack(padded_video_frames),
            "labels": torch.stack(padded_labels),
            "text_inputs": {
                "input_ids": torch.stack(text_inputs["input_ids"]),
                "attention_mask": torch.stack(text_inputs["attention_mask"]),
            },
            "text_queries": text_queries,
            "video_ids": video_ids,
            "gt_starts": torch.tensor(gt_starts),
            "gt_ends": torch.tensor(gt_ends),
        }
    
    def train_dataloader(self) -> DataLoader:
        """Get training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation data loader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test data loader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
