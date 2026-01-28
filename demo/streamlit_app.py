"""Streamlit demo for video moment retrieval."""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from ..models.video_moment_retrieval import VideoMomentRetrievalModel
from ..utils.device import get_device, set_seed
from ..utils.video import extract_frames, resize_frames, frames_to_tensor, get_video_info


class VideoMomentRetrievalDemo:
    """Demo application for video moment retrieval."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize demo.
        
        Args:
            model_path: Path to trained model checkpoint.
        """
        self.device = get_device()
        self.model = None
        self.model_path = model_path
        
        # Initialize model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the video moment retrieval model."""
        try:
            self.model = VideoMomentRetrievalModel(
                model_name="openai/clip-vit-base-patch32",
                max_frames=32,
                temporal_modeling=True,
            )
            
            if self.model_path and Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                st.success("Loaded trained model checkpoint!")
            else:
                st.info("Using pre-trained CLIP model (no fine-tuning)")
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None
    
    def process_video(
        self,
        video_file,
        query: str,
        max_frames: int = 32,
        top_k: int = 5,
    ) -> Tuple[List[np.ndarray], List[float], dict]:
        """Process video and retrieve moments.
        
        Args:
            video_file: Uploaded video file.
            query: Text query for moment retrieval.
            max_frames: Maximum number of frames to process.
            top_k: Number of top moments to retrieve.
            
        Returns:
            Tuple of (frames, scores, video_info).
        """
        if self.model is None:
            return [], [], {}
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        try:
            # Get video info
            video_info = get_video_info(video_path)
            
            # Extract frames
            frames = extract_frames(
                video_path,
                max_frames=max_frames,
                sampling_method="uniform",
            )
            
            if not frames:
                return [], [], video_info
            
            # Resize frames
            frames = resize_frames(frames, size=(224, 224))
            
            # Convert to tensor
            video_tensor = frames_to_tensor(frames)
            video_tensor = video_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            
            # Retrieve moments
            with torch.no_grad():
                results = self.model.retrieve_moments(
                    video_tensor,
                    query,
                    top_k=top_k,
                )
            
            # Get scores for all frames
            all_scores = results["all_scores"].cpu().numpy()
            
            return frames, all_scores, video_info
            
        except Exception as e:
            st.error(f"Error processing video: {e}")
            return [], [], {}
        
        finally:
            # Clean up temporary file
            Path(video_path).unlink(missing_ok=True)
    
    def create_visualization(
        self,
        frames: List[np.ndarray],
        scores: List[float],
        top_k: int = 5,
    ) -> List[np.ndarray]:
        """Create visualization with score overlays.
        
        Args:
            frames: List of video frames.
            scores: List of similarity scores.
            top_k: Number of top moments to highlight.
            
        Returns:
            List of frames with score overlays.
        """
        if not frames or not scores:
            return []
        
        # Get top-k frame indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Create visualization
        vis_frames = []
        for i, frame in enumerate(frames):
            # Create overlay
            overlay = frame.copy()
            
            # Add score text
            score = scores[i]
            cv2.putText(
                overlay,
                f"Score: {score:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            # Highlight top moments
            if i in top_indices:
                rank = list(top_indices).index(i) + 1
                cv2.putText(
                    overlay,
                    f"TOP {rank}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                # Add border
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1]-1, overlay.shape[0]-1), (0, 255, 0), 3)
            
            vis_frames.append(overlay)
        
        return vis_frames


def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Video Moment Retrieval Demo",
        page_icon="🎬",
        layout="wide",
    )
    
    st.title("🎬 Video Moment Retrieval Demo")
    st.markdown(
        "Upload a video and enter a text query to find the most relevant moments!"
    )
    
    # Initialize demo
    if "demo" not in st.session_state:
        st.session_state.demo = VideoMomentRetrievalDemo()
    
    demo = st.session_state.demo
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    max_frames = st.sidebar.slider(
        "Max Frames",
        min_value=8,
        max_value=64,
        value=32,
        help="Maximum number of frames to process from the video",
    )
    
    top_k = st.sidebar.slider(
        "Top K Moments",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top moments to highlight",
    )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Video upload
        video_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video file to analyze",
        )
        
        # Text query
        query = st.text_input(
            "Text Query",
            placeholder="e.g., 'person walking', 'dog running', 'car parking'",
            help="Describe the moment you want to find in the video",
        )
        
        # Process button
        process_button = st.button("🔍 Find Moments", type="primary")
    
    with col2:
        st.header("Results")
        
        if process_button and video_file and query:
            with st.spinner("Processing video..."):
                # Process video
                frames, scores, video_info = demo.process_video(
                    video_file,
                    query,
                    max_frames=max_frames,
                    top_k=top_k,
                )
                
                if frames and scores:
                    # Display video info
                    st.subheader("Video Information")
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
                    with col_info2:
                        st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
                    with col_info3:
                        st.metric("Frames", len(frames))
                    
                    # Create visualization
                    vis_frames = demo.create_visualization(frames, scores, top_k)
                    
                    # Display frames
                    st.subheader("Video Frames with Scores")
                    
                    # Create columns for frames
                    cols = st.columns(min(4, len(vis_frames)))
                    
                    for i, frame in enumerate(vis_frames):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            st.image(
                                frame,
                                caption=f"Frame {i+1}",
                                use_column_width=True,
                            )
                    
                    # Display score distribution
                    st.subheader("Score Distribution")
                    st.bar_chart(scores)
                    
                    # Display top moments
                    st.subheader("Top Moments")
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    
                    for rank, idx in enumerate(top_indices):
                        st.write(f"**#{rank+1}** Frame {idx+1} (Score: {scores[idx]:.3f})")
                        st.image(frames[idx], use_column_width=True)
                
                else:
                    st.error("Failed to process video. Please check your input.")
        
        elif process_button:
            if not video_file:
                st.error("Please upload a video file.")
            if not query:
                st.error("Please enter a text query.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Video Moment Retrieval** - Advanced Computer Vision Project | "
        "Built with PyTorch, CLIP, and Streamlit"
    )


if __name__ == "__main__":
    main()
