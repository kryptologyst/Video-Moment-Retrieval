"""Script to generate sample data for video moment retrieval."""

import json
import random
from pathlib import Path
from typing import List, Dict


def generate_sample_annotations(
    num_samples: int = 100,
    video_ids: List[str] = None,
    output_file: str = "data/annotations.json",
) -> None:
    """Generate sample annotations for video moment retrieval.
    
    Args:
        num_samples: Number of samples to generate.
        video_ids: List of video IDs. If None, generates random IDs.
        output_file: Output file path.
    """
    if video_ids is None:
        video_ids = [f"sample_video_{i:03d}.mp4" for i in range(20)]
    
    # Sample queries for different scenarios
    queries = [
        "person walking",
        "dog running",
        "car parking",
        "person sitting",
        "bird flying",
        "water flowing",
        "door opening",
        "person dancing",
        "cat sleeping",
        "ball bouncing",
        "person cooking",
        "tree swaying",
        "person reading",
        "dog barking",
        "car driving",
        "person jumping",
        "bird singing",
        "person laughing",
        "dog playing",
        "person writing",
    ]
    
    annotations = []
    
    for i in range(num_samples):
        # Random video ID
        video_id = random.choice(video_ids)
        
        # Random query
        query = random.choice(queries)
        
        # Random moment (assuming 30 frames per video)
        total_frames = 30
        start_frame = random.randint(0, total_frames - 5)
        end_frame = random.randint(start_frame + 1, min(start_frame + 10, total_frames - 1))
        
        annotation = {
            "id": i,
            "video_id": video_id,
            "query": query,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "description": f"Find the moment where {query}",
        }
        
        annotations.append(annotation)
    
    # Save annotations
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Generated {num_samples} sample annotations in {output_file}")


def create_sample_videos(
    output_dir: str = "data/videos",
    num_videos: int = 20,
    duration: float = 3.0,
    fps: int = 10,
) -> None:
    """Create sample videos for testing.
    
    Args:
        output_dir: Output directory for videos.
        num_videos: Number of videos to create.
        duration: Duration of each video in seconds.
        fps: Frames per second.
    """
    import cv2
    import numpy as np
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_videos):
        # Create video with random content
        video_path = output_path / f"sample_video_{i:03d}.mp4"
        
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Generate frames
        num_frames = int(duration * fps)
        
        for frame_idx in range(num_frames):
            # Create frame with random content
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some structure (simple patterns)
            if frame_idx % 10 < 5:
                # Add horizontal lines
                cv2.line(frame, (0, frame_idx * 10), (width, frame_idx * 10), (255, 255, 255), 2)
            else:
                # Add vertical lines
                cv2.line(frame, (frame_idx * 10, 0), (frame_idx * 10, height), (255, 255, 255), 2)
            
            # Add frame number
            cv2.putText(
                frame,
                f"Frame {frame_idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            out.write(frame)
        
        out.release()
        print(f"Created sample video: {video_path}")


def main():
    """Main function to generate sample data."""
    print("Generating sample data for video moment retrieval...")
    
    # Generate annotations
    generate_sample_annotations(
        num_samples=100,
        output_file="data/annotations.json",
    )
    
    # Create sample videos
    create_sample_videos(
        output_dir="data/videos",
        num_videos=20,
        duration=3.0,
        fps=10,
    )
    
    print("Sample data generation complete!")
    print("\nGenerated files:")
    print("- data/annotations.json (100 sample annotations)")
    print("- data/videos/ (20 sample videos)")
    print("\nYou can now run the training script or demo with this sample data.")


if __name__ == "__main__":
    main()
