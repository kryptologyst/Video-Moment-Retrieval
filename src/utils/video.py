"""Video processing utilities for moment retrieval."""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Union


def extract_frames(
    video_path: Union[str, Path],
    max_frames: int = 32,
    sampling_method: str = "uniform",
    target_fps: Optional[float] = None,
) -> List[np.ndarray]:
    """Extract frames from video with various sampling strategies.
    
    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to extract.
        sampling_method: Method for frame sampling ('uniform', 'random', 'keyframes').
        target_fps: Target FPS for frame extraction. If None, uses original FPS.
        
    Returns:
        List of extracted frames as numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if target_fps is None:
        target_fps = fps
    
    # Calculate frame indices based on sampling method
    if sampling_method == "uniform":
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    elif sampling_method == "random":
        frame_indices = np.sort(np.random.choice(total_frames, max_frames, replace=False))
    elif sampling_method == "keyframes":
        # Simple keyframe detection based on frame differences
        frame_indices = _extract_keyframes(cap, max_frames)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def _extract_keyframes(cap: cv2.VideoCapture, max_frames: int) -> List[int]:
    """Extract keyframes based on frame differences.
    
    Args:
        cap: OpenCV VideoCapture object.
        max_frames: Maximum number of keyframes to extract.
        
    Returns:
        List of keyframe indices.
    """
    frame_diffs = []
    prev_frame = None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    if not ret:
        return [0]
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        diff_score = np.mean(diff)
        frame_diffs.append((frame_idx, diff_score))
        
        prev_gray = gray
        frame_idx += 1
    
    # Sort by difference score and select top frames
    frame_diffs.sort(key=lambda x: x[1], reverse=True)
    keyframe_indices = [idx for idx, _ in frame_diffs[:max_frames]]
    keyframe_indices.sort()
    
    return keyframe_indices


def resize_frames(
    frames: List[np.ndarray],
    size: Tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_LINEAR,
) -> List[np.ndarray]:
    """Resize frames to target size.
    
    Args:
        frames: List of frames as numpy arrays.
        size: Target size (width, height).
        width: Target width.
        height: Target height.
        interpolation: OpenCV interpolation method.
        
    Returns:
        List of resized frames.
    """
    resized_frames = []
    for frame in frames:
        resized = cv2.resize(frame, size, interpolation=interpolation)
        resized_frames.append(resized)
    
    return resized_frames


def frames_to_tensor(
    frames: List[np.ndarray],
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Convert frames to PyTorch tensor.
    
    Args:
        frames: List of frames as numpy arrays.
        normalize: Whether to normalize frames.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        
    Returns:
        PyTorch tensor of shape (T, C, H, W).
    """
    # Convert to PIL Images first
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    # Convert to tensors
    tensor_frames = []
    for pil_frame in pil_frames:
        # Convert PIL to tensor
        frame_tensor = torch.from_numpy(np.array(pil_frame)).float()
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        if normalize:
            frame_tensor = frame_tensor / 255.0
            mean_tensor = torch.tensor(mean).view(3, 1, 1)
            std_tensor = torch.tensor(std).view(3, 1, 1)
            frame_tensor = (frame_tensor - mean_tensor) / std_tensor
        
        tensor_frames.append(frame_tensor)
    
    # Stack frames
    video_tensor = torch.stack(tensor_frames, dim=0)  # (T, C, H, W)
    return video_tensor


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: float = 30.0,
) -> None:
    """Create video from frames.
    
    Args:
        frames: List of frames as numpy arrays.
        output_path: Output video path.
        fps: Frames per second.
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def get_video_info(video_path: Union[str, Path]) -> dict:
    """Get video information.
    
    Args:
        video_path: Path to video file.
        
    Returns:
        Dictionary with video information.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }
    
    cap.release()
    return info
