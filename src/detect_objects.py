from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "input" / "seq38" / "Images"
OUTPUT_DIR = Path("outputs/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Aerial relevant COCO classes: person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
TARGET_CLASSES = [0, 1, 2, 3, 5, 7]

print("Loading YOLO11s model...")

model = YOLO("yolo11s.pt")

print("YOLO model loaded.")

images = sorted(INPUT_DIR.glob("*.png"))

print(f"Found {len(images)} images.")

for i, image_path in enumerate(images, start=1):
    print(f"[{i}/{len(images)}] Processing {image_path.name}...")

    results = model.predict(
        source=str(image_path),
        device="cpu",
        conf=0.30,
        imgsz=1280,
        classes=TARGET_CLASSES,
        save=True,
        project=str(OUTPUT_DIR),
        name="annotated",
        exist_ok=True,
        verbose=False,
    )

print("All object detections completed!")
print(f"Results saved in: runs/detect/outputs/detections/annotated")