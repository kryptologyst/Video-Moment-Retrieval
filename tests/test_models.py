"""Test for video moment retrieval models."""

import pytest
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.video_moment_retrieval import VideoMomentRetrievalModel, TemporalAttention
from utils.device import get_device, set_seed


class TestTemporalAttention:
    """Test temporal attention module."""
    
    def test_temporal_attention_forward(self):
        """Test temporal attention forward pass."""
        batch_size, seq_len, embed_dim = 2, 16, 512
        attention = TemporalAttention(embed_dim, num_heads=8)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        output = attention(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_temporal_attention_with_mask(self):
        """Test temporal attention with attention mask."""
        batch_size, seq_len, embed_dim = 2, 16, 512
        attention = TemporalAttention(embed_dim, num_heads=8)
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        mask = torch.ones(batch_size, seq_len)
        mask[:, -4:] = 0  # Mask last 4 frames
        
        output = attention(x, attention_mask=mask)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestVideoMomentRetrievalModel:
    """Test video moment retrieval model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = VideoMomentRetrievalModel(
            model_name="openai/clip-vit-base-patch32",
            max_frames=16,
            temporal_modeling=True,
        )
        
        assert model.max_frames == 16
        assert model.temporal_modeling == True
        assert model.clip_model is not None
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = VideoMomentRetrievalModel(
            model_name="openai/clip-vit-base-patch32",
            max_frames=8,
            temporal_modeling=True,
        )
        
        batch_size, num_frames = 2, 8
        video_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        text_inputs = {
            "input_ids": torch.randint(0, 1000, (batch_size, 77)),
            "attention_mask": torch.ones(batch_size, 77),
        }
        
        outputs = model(video_frames, text_inputs)
        
        assert "similarity_scores" in outputs
        assert "moment_scores" in outputs
        assert "video_embeddings" in outputs
        assert "text_embeddings" in outputs
        
        assert outputs["similarity_scores"].shape == (batch_size, num_frames)
        assert outputs["moment_scores"].shape == (batch_size, 1)
    
    def test_model_retrieve_moments(self):
        """Test moment retrieval functionality."""
        model = VideoMomentRetrievalModel(
            model_name="openai/clip-vit-base-patch32",
            max_frames=8,
            temporal_modeling=True,
        )
        
        batch_size, num_frames = 1, 8
        video_frames = torch.randn(batch_size, num_frames, 3, 224, 224)
        query = "person walking"
        
        results = model.retrieve_moments(video_frames, query, top_k=3)
        
        assert "top_indices" in results
        assert "top_scores" in results
        assert "all_scores" in results
        
        assert len(results["top_indices"]) == 3
        assert len(results["top_scores"]) == 3
        assert len(results["all_scores"]) == num_frames


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise errors
        assert True


if __name__ == "__main__":
    pytest.main([__file__])
