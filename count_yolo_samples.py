import os
from tqdm import tqdm

def count_samples(base_dir, split):
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        print(f"{split_dir} does not exist.")
        return {}
    class_counts = {}
    class_dirs = [d for d in sorted(os.listdir(split_dir)) if os.path.isdir(os.path.join(split_dir, d))]
    for class_name in tqdm(class_dirs, desc=f"Counting {split} samples", colour='cyan'):
        class_path = os.path.join(split_dir, class_name)
        count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
        class_counts[class_name] = count
    return class_counts

def print_counts(title, counts):
    print(f"\n\033[1m=== {title} ===\033[0m")
    total = 0
    for class_name, count in counts.items():
        print(f"  \033[94m{class_name:<12}\033[0m: {count:>7} samples")
        total += count
    print(f"  \033[92mTotal        : {total:>7} samples\033[0m")

if __name__ == "__main__":
    yolo_base = "outputs/cnn_results/250901_013612/fold_BallenyIslands2015_noise_0.25/yolo_dataset"
    for data_type in ["images", "labels"]:
        print(f"\n{'='*10} \033[1m{data_type.upper()}\033[0m {'='*10}")
        for split in ["train", "valid", "test"]:
            counts = count_samples(os.path.join(yolo_base, data_type), split)
            print_counts(f"{split.upper()} partition", counts)
