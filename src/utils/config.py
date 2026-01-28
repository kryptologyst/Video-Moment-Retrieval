"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the video moment retrieval project."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> DictConfig:
        """Load configuration from file.
        
        Returns:
            DictConfig: Loaded configuration.
        """
        if self.config_path.exists():
            return OmegaConf.load(self.config_path)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration.
        
        Returns:
            DictConfig: Default configuration.
        """
        default_config = {
            "model": {
                "name": "clip",
                "pretrained": "openai/clip-vit-base-patch32",
                "temporal_modeling": True,
                "max_frames": 32,
                "frame_sampling": "uniform",
            },
            "data": {
                "video_dir": "data/videos",
                "annotation_file": "data/annotations.json",
                "batch_size": 8,
                "num_workers": 4,
                "video_fps": 1.0,
                "image_size": 224,
            },
            "training": {
                "epochs": 100,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "warmup_steps": 1000,
                "gradient_accumulation_steps": 1,
                "mixed_precision": True,
                "save_every": 10,
                "eval_every": 5,
            },
            "evaluation": {
                "metrics": ["r@1", "r@5", "r@10", "mAP", "temporal_consistency"],
                "iou_threshold": 0.5,
                "temporal_window": 5,
            },
            "device": {
                "device": "auto",
                "mixed_precision_dtype": "auto",
            },
            "logging": {
                "log_dir": "logs",
                "tensorboard": True,
                "wandb": False,
                "project_name": "video-moment-retrieval",
            },
            "paths": {
                "checkpoint_dir": "checkpoints",
                "output_dir": "outputs",
                "asset_dir": "assets",
            },
        }
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self._config, key, default=default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        OmegaConf.set(self._config, key, value)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates.
        """
        self._config.update(updates)
    
    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self._config, save_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return OmegaConf.to_container(self._config, resolve=True)
    
    @property
    def config(self) -> DictConfig:
        """Get the underlying DictConfig object.
        
        Returns:
            DictConfig: Configuration object.
        """
        return self._config
