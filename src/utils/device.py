"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device with fallback order: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_mixed_precision_dtype() -> torch.dtype:
    """Get the appropriate mixed precision dtype based on device.
    
    Returns:
        torch.dtype: The appropriate dtype for mixed precision training.
    """
    device = get_device()
    if device.type == "cuda":
        # Use bfloat16 for newer GPUs, fallback to float16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device.type == "mps":
        # MPS supports float16
        return torch.float16
    else:
        # CPU doesn't benefit from mixed precision
        return torch.float32


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get the model size in megabytes.
    
    Args:
        model: PyTorch model.
        
    Returns:
        float: Model size in MB.
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb
