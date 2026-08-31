from pathlib import Path
from ultralytics import YOLO

INPUT_DIR = Path("data/input/seq38/Images")
OUTPUT_DIR = Path("outputs/detections")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded.")

images = sorted(INPUT_DIR.glob("*.png"))

print(f"Found {len(images)} images.")

for i, image_path in enumerate(images, start=1):
    print(f"[{i}/{len(images)}] Processing {image_path.name}...")

    results = model.predict(
        source=str(image_path),
        device="cpu",
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="annotated",
        exist_ok=True,
        verbose=False,
    )

print("All object detections completed!")
print(f"Results saved in: {OUTPUT_DIR}")