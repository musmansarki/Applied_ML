import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
import joblib
from pathlib import Path
import os
from PIL import Image
import io
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def load_and_resize_image(file_path, target_size=(64, 64)):
    """Load and resize an image to target size."""
    arr = np.load(file_path)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img) / 255.0

def load_data(folder, batch_size=1000, target_size=(64, 64)):
    """Load and preprocess data from a folder of .npy files in batches."""
    data = []
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_data = []
        for file in batch_files:
            arr = load_and_resize_image(os.path.join(folder, file), target_size)
            batch_data.append(arr.flatten())
        data.extend(batch_data)
        print(f"Processed {min(i + batch_size, len(files))}/{len(files)} files")
    
    return np.array(data)

def main():
    base_path = Path(__file__).resolve().parent
    train_path = base_path / 'data' / 'ProcessedResizedNorm' / 'train'
    models_path = base_path / 'models'
    models_path.mkdir(exist_ok=True)

    cat_folder = train_path / 'Cat'
    dog_folder = train_path / 'Dog'

    if not cat_folder.exists():
        raise FileNotFoundError(f"Missing folder: {cat_folder}")
    if not dog_folder.exists():
        raise FileNotFoundError(f"Missing folder: {dog_folder}")

    print("Loading training data...")
    X_cat = load_data(cat_folder)
    X_dog = load_data(dog_folder)

    y_cat = np.zeros(len(X_cat))
    y_dog = np.ones(len(X_dog))

    X = np.vstack([X_cat, X_dog])
    y = np.concatenate([y_cat, y_dog])

    n_features = X.shape[1]
    n_components = min(n_features, 100)  
    print(f"Original feature dimension: {n_features}")
    print(f"Using {n_components} PCA components")

    print("Training Random Forest model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', IncrementalPCA(n_components=n_components, batch_size=1000)),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            verbose=1
        ))
    ])

    pipeline.fit(X, y)

    pca = pipeline.named_steps['pca']
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Total explained variance: {explained_variance:.2%}")

    # Evaluate model performance
    y_pred = pipeline.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Check for over/underfitting
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.2%}")

    # Compare to baseline
    baseline_accuracy = 0.5  # Assuming a simple baseline
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Improvement over baseline: {accuracy - baseline_accuracy:.2%}")

    # Measure of variance
    pca = pipeline.named_steps['pca']
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Total explained variance for Random Forest: {explained_variance:.2%}")

    output_path = models_path / 'rf_pipeline.joblib'
    joblib.dump(pipeline, output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()


