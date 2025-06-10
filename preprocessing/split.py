import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Use absolute path from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR     = PROJECT_ROOT / 'data' / 'PetImages'
OUTPUT_DIR   = PROJECT_ROOT / 'data' / 'Processed'
CATEGORIES   = ['Cat', 'Dog']
SEED         = 42

# Gather paths and labels
paths = []
labels = []
for cls in CATEGORIES:
    for img in (BASE_DIR / cls).glob('*'):
        if img.is_file():
            paths.append(img)
            labels.append(cls)

# train (80%) vs. temp (20%)
train_p, temp_p, train_l, temp_l = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# validation (10%) vs. test (10%) from temp
val_p, test_p, val_l, test_l = train_test_split(
    temp_p, temp_l, test_size=0.5, stratify=temp_l, random_state=SEED
)

splits = {
    'train': (train_p, train_l),
    'val':   (val_p,   val_l),
    'test':  (test_p,  test_l),
}

for split_name, (split_paths, split_labels) in splits.items():
    for img_path, label in zip(split_paths, split_labels):
        dest = OUTPUT_DIR / split_name / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest / img_path.name)

for name, (split_paths, _) in splits.items():
    print(f"{name.capitalize()}: {len(split_paths)} images")
