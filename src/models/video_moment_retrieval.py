"""Advanced video moment retrieval models."""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from transformers.modeling_outputs import BaseModelOutput

from ..utils.device import get_device


class TemporalAttention(nn.Module):
    """Temporal attention module for video understanding."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_length: int = 32,
    ):
        """Initialize temporal attention.
        
        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            max_length: Maximum sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
            attention_mask: Attention mask.
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Add positional encoding
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        output = self.out_proj(attn_output)
        
        return output


class VideoMomentRetrievalModel(nn.Module):
    """Advanced video moment retrieval model with temporal modeling."""
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        max_frames: int = 32,
        temporal_modeling: bool = True,
        freeze_clip: bool = False,
    ):
        """Initialize the model.
        
        Args:
            model_name: CLIP model name.
            max_frames: Maximum number of frames.
            temporal_modeling: Whether to use temporal modeling.
            freeze_clip: Whether to freeze CLIP parameters.
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_frames = max_frames
        self.temporal_modeling = temporal_modeling
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Get embedding dimensions
        self.vision_embed_dim = self.clip_model.config.vision_config.hidden_size
        self.text_embed_dim = self.clip_model.config.text_config.hidden_size
        
        # Temporal modeling
        if temporal_modeling:
            self.temporal_attention = TemporalAttention(
                embed_dim=self.vision_embed_dim,
                num_heads=8,
                max_length=max_frames,
            )
            self.temporal_proj = nn.Linear(self.vision_embed_dim, self.text_embed_dim)
        
        # Cross-modal fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.text_embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # Moment prediction head
        self.moment_head = nn.Sequential(
            nn.Linear(self.text_embed_dim, self.text_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.text_embed_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def encode_video(self, video_frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames.
        
        Args:
            video_frames: Video frames tensor of shape (batch_size, num_frames, C, H, W).
            
        Returns:
            Video embeddings of shape (batch_size, num_frames, embed_dim).
        """
        batch_size, num_frames = video_frames.shape[:2]
        
        # Reshape for CLIP processing
        video_frames_flat = video_frames.view(-1, *video_frames.shape[2:])
        
        # Get CLIP vision embeddings
        vision_outputs = self.clip_model.vision_model(
            pixel_values=video_frames_flat,
            return_dict=True,
        )
        vision_embeddings = vision_outputs.last_hidden_state  # (batch_size * num_frames, seq_len, embed_dim)
        
        # Pool the sequence dimension (CLIP uses CLS token)
        vision_embeddings = vision_embeddings[:, 0, :]  # (batch_size * num_frames, embed_dim)
        
        # Reshape back to video format
        vision_embeddings = vision_embeddings.view(batch_size, num_frames, -1)
        
        # Apply temporal modeling
        if self.temporal_modeling:
            vision_embeddings = self.temporal_attention(vision_embeddings)
            vision_embeddings = self.temporal_proj(vision_embeddings)
        
        return vision_embeddings
    
    def encode_text(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode text queries.
        
        Args:
            text_inputs: Text inputs from CLIP processor.
            
        Returns:
            Text embeddings of shape (batch_size, embed_dim).
        """
        text_outputs = self.clip_model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            return_dict=True,
        )
        text_embeddings = text_outputs.last_hidden_state  # (batch_size, seq_len, embed_dim)
        
        # Pool the sequence dimension
        text_embeddings = text_embeddings[:, 0, :]  # (batch_size, embed_dim)
        
        return text_embeddings
    
    def forward(
        self,
        video_frames: torch.Tensor,
        text_inputs: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            video_frames: Video frames tensor.
            text_inputs: Text inputs from CLIP processor.
            return_attention: Whether to return attention weights.
            
        Returns:
            Dictionary containing predictions and optional attention weights.
        """
        # Encode video and text
        video_embeddings = self.encode_video(video_frames)  # (batch_size, num_frames, embed_dim)
        text_embeddings = self.encode_text(text_inputs)  # (batch_size, embed_dim)
        
        # Expand text embeddings for cross-attention
        text_embeddings_expanded = text_embeddings.unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # Cross-attention between text and video
        attn_output, attn_weights = self.cross_attention(
            query=text_embeddings_expanded,
            key=video_embeddings,
            value=video_embeddings,
        )
        
        # Compute similarity scores for each frame
        similarity_scores = F.cosine_similarity(
            text_embeddings_expanded, video_embeddings, dim=-1
        )  # (batch_size, num_frames)
        
        # Predict moment relevance
        moment_scores = self.moment_head(attn_output.squeeze(1))  # (batch_size, 1)
        
        outputs = {
            "similarity_scores": similarity_scores,
            "moment_scores": moment_scores,
            "video_embeddings": video_embeddings,
            "text_embeddings": text_embeddings,
        }
        
        if return_attention:
            outputs["attention_weights"] = attn_weights
        
        return outputs
    
    def retrieve_moments(
        self,
        video_frames: torch.Tensor,
        text_query: str,
        top_k: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """Retrieve top-k moments for a text query.
        
        Args:
            video_frames: Video frames tensor.
            text_query: Text query string.
            top_k: Number of top moments to retrieve.
            
        Returns:
            Dictionary containing retrieved moments and scores.
        """
        # Process text query
        text_inputs = self.clip_processor(
            text=[text_query],
            return_tensors="pt",
            padding=True,
        )
        
        # Move to same device as video frames
        device = video_frames.device
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(video_frames, text_inputs)
        
        # Get top-k moments
        similarity_scores = outputs["similarity_scores"][0]  # (num_frames,)
        top_indices = torch.topk(similarity_scores, top_k).indices
        
        return {
            "top_indices": top_indices,
            "top_scores": similarity_scores[top_indices],
            "all_scores": similarity_scores,
        }
