import os
from pathlib import Path
from PIL import Image

# Use absolute path from project root
project_root = Path(__file__).resolve().parent.parent
base_dir = project_root / 'data' / 'PetImages'

categories = ['Cat', 'Dog']
corrupt_dir = base_dir / 'Corrupt'

corrupt_dir.mkdir(parents=True, exist_ok=True)
for cat in categories:
    (corrupt_dir / cat).mkdir(exist_ok=True)

corrupt_files = []

for cat in categories:
    folder = base_dir / cat
    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue
    for img_path in folder.iterdir():
        if not img_path.is_file():
            continue
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception:
            corrupt_files.append(str(img_path))
            (corrupt_dir / cat / img_path.name).parent.mkdir(exist_ok=True)
            img_path.rename(corrupt_dir / cat / img_path.name)

# Write log
log_path = base_dir / 'corrupt_files.txt'
with open(log_path, 'w') as f:
    for p in corrupt_files:
        f.write(p + '\n')

print(f"✔ Found {len(corrupt_files)} corrupt images. Logs → {log_path}")
