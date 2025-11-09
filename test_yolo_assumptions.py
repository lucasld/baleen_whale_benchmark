#!/usr/bin/env python3
"""
Quick verification script for YOLOv12 assumptions on cluster.
Tests model availability, image size compatibility, rectangular training support,
and basic functionality without training on actual data.

Usage: python test_yolo_assumptions.py
"""

import os
import sys
import torch

# Suppress Ultralytics logging for clean output
os.environ["RICH_PROGRESS_BAR"] = "0"
os.environ["ULTRALYTICS_QUIET"] = "1"

# Set models directory
from ultralytics import settings
settings.update({"weights_dir": "models"})

try:
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    print("✓ Ultralytics YOLO imported successfully")
    print(f"✓ Models will be downloaded to: {settings.get('weights_dir')}")
except ImportError as e:
    print(f"✗ Failed to import Ultralytics: {e}")
    sys.exit(1)

def test_model_availability():
    """Test if YOLOv12 models are available for download/load."""
    print("\n=== Testing YOLOv12 Model Availability ===")
    models = ['yolov12n.pt', 'yolov12m.pt', 'yolov12x.pt']

    for model_name in models:
        try:
            print(f"Loading {model_name}...")
            model = YOLO(model_name)
            print(f"✓ {model_name} loaded successfully")
            # Check if it's actually YOLOv12
            if hasattr(model.model, 'yaml') and 'yolov12' in str(model.model.yaml).lower():
                print(f"✓ Confirmed {model_name} is YOLOv12")
            else:
                print(f"⚠ {model_name} may not be YOLOv12 - check model info")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")

def test_image_sizes():
    """Test image size compatibility with YOLOv12."""
    print("\n=== Testing Image Size Compatibility ===")
    model = YOLO('yolov12n.pt')  # Use smallest model for speed

    test_sizes = [192, 256, 320, 512]

    for imgsz in test_sizes:
        try:
            print(f"Testing imgsz={imgsz}...")

            # Check stride compatibility (must be divisible by 32 for YOLO)
            stride = 32  # YOLOv12 max stride
            if imgsz % stride != 0:
                print(f"⚠ imgsz={imgsz} not divisible by {stride} - may be rounded up by Ultralytics")
            else:
                print(f"✓ imgsz={imgsz} compatible with stride {stride}")

            # Test model creation with this size (basic initialization)
            # We won't actually train, just check if the model can be created
            try:
                # Create a dummy input tensor to test forward pass
                dummy_input = torch.randn(1, 3, imgsz, imgsz)
                with torch.no_grad():
                    _ = model.model(dummy_input)
                print(f"✓ Model forward pass works with imgsz={imgsz}")
            except Exception as e:
                print(f"✗ Model forward pass failed with imgsz={imgsz}: {e}")

        except Exception as e:
            print(f"✗ Error testing imgsz={imgsz}: {e}")

def test_rectangular_training():
    """Test rectangular training support."""
    print("\n=== Testing Rectangular Training Support ===")
    model = YOLO('yolov12n.pt')

    # Test rect=True with different img sizes
    test_configs = [
        {'imgsz': 256, 'rect': True},  # Our main size
        {'imgsz': 192, 'rect': True},  # Lower bound
        {'imgsz': 320, 'rect': True},  # Higher resolution
    ]

    for config in test_configs:
        try:
            print(f"Testing rect training: imgsz={config['imgsz']}, rect={config['rect']}")

            # Check if Ultralytics supports rect parameter
            # We'll simulate by checking model creation
            # In practice, rect=True is handled during training, not model loading
            print("✓ Ultralytics typically supports rect=True for rectangular training")
            print("  (Full validation requires actual training loop with dataset)")

        except Exception as e:
            print(f"✗ Error with rect config {config}: {e}")

def test_batch_sizes():
    """Test if batch sizes work (basic memory check)."""
    print("\n=== Testing Batch Size Compatibility ===")
    model = YOLO('yolov12n.pt')

    test_batches = [16, 32, 64]
    imgsz = 256  # Our main size

    for batch in test_batches:
        try:
            print(f"Testing batch_size={batch} with imgsz={imgsz}")

            # Create dummy batch
            dummy_batch = torch.randn(batch, 3, imgsz, imgsz)

            # Test forward pass
            with torch.no_grad():
                _ = model.model(dummy_batch)

            print(f"✓ Batch size {batch} works (memory-wise)")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠ Batch size {batch} may cause OOM - consider smaller batch")
            else:
                print(f"✗ Batch size {batch} failed: {e}")
        except Exception as e:
            print(f"✗ Error testing batch {batch}: {e}")

def test_gpu_availability():
    """Check GPU availability."""
    print("\n=== Checking GPU Availability ===")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ CUDA available with {device_count} GPU(s)")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    else:
        print("⚠ No CUDA GPUs detected - will run on CPU")

def main():
    """Run all tests."""
    print("YOLOv12 Assumptions Verification Script")
    print("=" * 50)

    test_gpu_availability()
    test_model_availability()
    test_image_sizes()
    test_rectangular_training()
    test_batch_sizes()

    print("\n" + "=" * 50)
    print("Verification complete!")
    print("\nNotes:")
    print("- This script tests basic functionality without actual training.")
    print("- For full rect=True validation, run a short training job.")
    print("- Memory tests are approximate; actual usage depends on dataset and augmentations.")

if __name__ == "__main__":
    main()
