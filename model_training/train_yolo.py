from ultralytics import YOLO
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = BASE_DIR / "datasets" / "kvasir_yolo" / "data.yaml"

def train():
    # Load a model
    # Use yolo11n.pt if available, otherwise yolov8n.pt
    # Ultralytics package usually handles download
    model = YOLO('yolo11n.pt')  # build a new model from YAML or load a pretrained model

    # Train the model
    results = model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=4, # Reduced to avoid OOM
        workers=0, # Reduced to avoid OOM
        project=str(BASE_DIR / 'runs' / 'detect'),
        name='train',
        exist_ok=True, # Overwrite existing experiment
        device='cpu' # GPU driver mismatch detected, forcing CPU.
    )

    # Validate
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")

    # Export? The .pt file is already saved.

if __name__ == "__main__":
    train()
