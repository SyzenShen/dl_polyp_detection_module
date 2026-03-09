import json
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_IMAGES_DIR = BASE_DIR / "Kvasir-SEG" / "images"
SOURCE_JSON_PATH = BASE_DIR / "Kvasir-SEG" / "kavsir_bboxes.json"
DATASET_DIR = BASE_DIR / "datasets" / "kvasir_yolo"

def setup_directories():
    """Create necessary directories for YOLO format."""
    for split in ['train', 'val', 'test']:
        (DATASET_DIR / 'images' / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"Directories created at {DATASET_DIR}")

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert bounding box to YOLO format (class x_center y_center width height)."""
    xmin = bbox['xmin']
    ymin = bbox['ymin']
    xmax = bbox['xmax']
    ymax = bbox['ymax']
    
    # Calculate center and width/height
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Clip values to [0, 1] just in case
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def prepare_data():
    if not SOURCE_IMAGES_DIR.exists():
        print(f"Error: Source images directory not found at {SOURCE_IMAGES_DIR}")
        return
    if not SOURCE_JSON_PATH.exists():
        print(f"Error: Source JSON not found at {SOURCE_JSON_PATH}")
        return

    print("Loading JSON data...")
    with open(SOURCE_JSON_PATH, 'r') as f:
        data = json.load(f)

    # Filter entries that have matching image files
    valid_entries = []
    for img_id, info in data.items():
        img_path = SOURCE_IMAGES_DIR / f"{img_id}.jpg"
        if img_path.exists():
            valid_entries.append(img_id)
        else:
            # Try checking if it's png? The ls showed jpg.
            pass
            
    print(f"Found {len(valid_entries)} valid images out of {len(data)} entries in JSON.")

    # Split dataset
    # 80% train, 10% val, 10% test
    train_ids, test_ids = train_test_split(valid_entries, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    setup_directories()

    for split, ids in splits.items():
        print(f"Processing {split} set ({len(ids)} images)...")
        for img_id in tqdm(ids):
            info = data[img_id]
            img_width = info['width']
            img_height = info['height']
            
            # Copy image
            src_img = SOURCE_IMAGES_DIR / f"{img_id}.jpg"
            dst_img = DATASET_DIR / 'images' / split / f"{img_id}.jpg"
            shutil.copy2(src_img, dst_img)
            
            # Generate label file
            label_lines = []
            if 'bbox' in info:
                for bbox in info['bbox']:
                    if bbox['label'] == 'polyp':
                        yolo_line = convert_bbox_to_yolo(bbox, img_width, img_height)
                        label_lines.append(yolo_line)
            
            dst_label = DATASET_DIR / 'labels' / split / f"{img_id}.txt"
            with open(dst_label, 'w') as f:
                f.write('\n'.join(label_lines))

    print("Data preparation complete!")
    
    # Create data.yaml
    yaml_content = f"""
path: {DATASET_DIR.absolute()} # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test: images/test # test images (optional)

names:
  0: polyp
"""
    with open(DATASET_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Created data.yaml at {DATASET_DIR / 'data.yaml'}")

if __name__ == "__main__":
    prepare_data()
