
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = (
    PROJECT_DIR
    / "weed detection -bounding boxes-.v7i.yolov8"
    / "data.yaml"
)

OUTPUT_DIR = PROJECT_DIR / "training_results"


def main():
    print("Loading pretrained YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")

    print("Starting weed detection training on CPU...")
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=416,
        batch=4,
        device="cpu",
        workers=0,
        patience=10,
        project=str(OUTPUT_DIR),
        name="weed_model_v2",
        exist_ok=True,
        plots=True,
    )

    print("\nTraining finished!")
    print("Best model:")
    print(OUTPUT_DIR / "weed_model_v2" / "weights" / "best.pt")


if __name__ == "__main__":
    main()
