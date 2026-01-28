#!/usr/bin/env python3
"""Setup script for video moment retrieval project."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Video Moment Retrieval Project")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Dependency installation failed. Please install manually:")
        print("   pip install -r requirements.txt")
    
    # Generate sample data
    if not run_command("python scripts/generate_sample_data.py", "Generating sample data"):
        print("⚠️  Sample data generation failed. You can run it manually later:")
        print("   python scripts/generate_sample_data.py")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Tests failed. You can run them manually later:")
        print("   python -m pytest tests/ -v")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run the demo: streamlit run demo/streamlit_app.py")
    print("2. Train a model: python scripts/train.py --config configs/small.yaml")
    print("3. Evaluate model: python scripts/evaluate.py --model checkpoints/best_model.pt")
    print("4. Explore the notebook: jupyter notebook notebooks/demo.ipynb")
    
    print("\n📁 Project structure:")
    print("├── src/                    # Source code")
    print("├── configs/               # Configuration files")
    print("├── scripts/               # Executable scripts")
    print("├── demo/                  # Demo applications")
    print("├── data/                  # Data directory")
    print("├── tests/                 # Unit tests")
    print("├── notebooks/             # Jupyter notebooks")
    print("└── README.md              # Documentation")


if __name__ == "__main__":
    main()
