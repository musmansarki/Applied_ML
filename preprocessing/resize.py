import sys
from pathlib import Path
from PIL import Image
import numpy as np

def resize_and_normalize(input_root, output_root, size=(224, 224)):
    """
    Resize all images, normalize pixel values to [0,1]
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    for split in ["train", "val", "test"]:
        for label in ["Cat", "Dog"]:
            src_dir = input_root / split / label
            dst_dir = output_root / split / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing {split}/{label} ...")
            for img_path in src_dir.glob("*"):
                if not img_path.is_file():
                    continue
                try:
                    # Load image, convert, resize
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = img.resize(size, Image.BILINEAR)
                        # Convert to numpy array and normalize to [0,1]
                        arr = np.array(img, dtype=np.float32) / 255.0
                        # Save normalized array
                        out_path = dst_dir / (img_path.stem + ".npy")
                        np.save(out_path, arr)
                except Exception as e:
                    print(f"  ✗ Failed: {img_path.name} ({e})")

if __name__ == "__main__":
    # Use absolute path from project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "Processed"
    out_dir  = project_root / "data" / "ProcessedResizedNorm"
    if len(sys.argv) >= 3:
        data_dir = Path(sys.argv[1])
        out_dir  = Path(sys.argv[2])
    resize_and_normalize(data_dir, out_dir, size=(224, 224))
    print("✔ Done! Normalized arrays are in:", out_dir)
