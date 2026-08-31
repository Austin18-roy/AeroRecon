from pathlib import Path

from PIL import Image
from transformers import pipeline

INPUT_DIR = Path("data/input/seq38/Images")
OUTPUT_DIR = Path("outputs/depth")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Depth Anything V2 Small...")

pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=-1,
)

print("Model loaded.")

images = sorted(INPUT_DIR.glob("*.png"))

print(f"Found {len(images)} images.")

for i, image_path in enumerate(images, start=1):
    print(f"[{i}/{len(images)}] Processing {image_path.name}...")

    image = Image.open(image_path).convert("RGB")
    result = pipe(image)

    depth_image = result["depth"]

    output_path = OUTPUT_DIR / f"depth_{image_path.stem}.png"
    depth_image.save(output_path)

    print(f"    Saved: {output_path}")

print("All depth maps generated successfully!")