import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.yolo.aug import NoiseMix

class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, overrides=None, _callbacks=None, custom_transforms=None):
        super().__init__(overrides, _callbacks)
        self.custom_transforms = custom_transforms
        print(f"[DEBUG] CustomDetectionTrainer initialized with {len(self.custom_transforms) if self.custom_transforms else 0} custom transforms")

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Override build_dataset to inject custom Albumentations transforms.
        """
        # Use the parent's logic to build the dataset correctly
        dataset = super().build_dataset(img_path, mode, batch)
        
        # If this is training and we have custom transforms, we wrap the dataset's transform pipeline
        if mode == "train" and self.custom_transforms:
            print(f"[DEBUG] Injecting {len(self.custom_transforms)} transforms into {mode} dataset")
            
            # In Ultralytics 8.3.x, the dataset has a .transforms attribute that is a Compose object
            old_transforms = dataset.transforms
            
            def wrapped_transform(labels):
                # 1. Apply our custom NoiseMix first
                # NoiseMix is an ImageOnlyTransform, so we pass 'image' and get 'image' back
                for t in self.custom_transforms:
                    labels["img"] = t(image=labels["img"])["image"]
                
                # 2. Apply standard YOLO transforms (mosaic, mixup, etc.)
                if old_transforms:
                    return old_transforms(labels)
                return labels
            
            dataset.transforms = wrapped_transform
            
        return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to CNN run folder")
    parser.add_argument("--train_fold", type=str, required=True, help="Fold name (e.g. fold_BallenyIslands2015_noise_0.25)")
    args = parser.parse_args()

    run_root = Path(args.run_dir)
    train_fold = run_root / args.train_fold
    ds_root = train_fold / "yolo_dataset"
    data_yaml = train_fold / "yolo" / "runs_det" / "C2" / "data_trainval.yaml"
    
    # Fallback if the specific C2 yaml doesn't exist yet
    if not data_yaml.exists():
        # Try to find any data_trainval.yaml in the fold
        yamls = list(train_fold.rglob("data_trainval.yaml"))
        if yamls:
            data_yaml = yamls[0]
        else:
            print(f"Error: Could not find data_trainval.yaml in {train_fold}")
            sys.exit(1)

    print(f"Using dataset: {ds_root}")
    print(f"Using YAML: {data_yaml}")

    # Collect noise paths (negatives)
    labels_dir = ds_root / "labels" / "train"
    images_dir = ds_root / "images" / "train"
    noise_paths = []
    for lp in sorted(labels_dir.glob("*.txt")):
        if lp.stat().st_size == 0:
            stem = lp.stem
            for ext in [".png", ".jpg", ".jpeg"]:
                img_p = images_dir / f"{stem}{ext}"
                if img_p.exists():
                    noise_paths.append(img_p)
                    break
    
    if not noise_paths:
        print("Error: No noise samples found in training set.")
        sys.exit(1)
    
    print(f"Found {len(noise_paths)} noise samples.")

    # Initialize NoiseMix
    # We use p=1.0 and a high alpha for the test so we can clearly see if it's working in logs/output
    my_noise_mix = NoiseMix(noise_paths=noise_paths, alpha=0.5, p=1.0)

    # Training Overrides
    overrides = {
        "model": str(project_root / "models" / "yolo12n.pt"), # Use local YOLOv12 model
        "data": str(data_yaml),
        "epochs": 2,           # Just 2 epochs to verify it runs
        "imgsz": 320,
        "batch": 8,
        "device": 0,           # Use GPU
        "project": "test/runs",
        "name": "noise_mix_test",
        "exist_ok": True,
        "plots": True,
    }

    print("Starting Custom Trainer Test...")
    # Properly merge overrides with default configuration to avoid AttributeError
    cfg = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
    trainer = CustomDetectionTrainer(overrides=cfg, custom_transforms=[my_noise_mix])
    trainer.train()
    print("Test Completed Successfully!")

if __name__ == "__main__":
    main()
