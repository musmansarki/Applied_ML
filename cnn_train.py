import os
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Custom transform for Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std})"

class NPYImageDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        self.labels = []
        self.augment = augment
        for label, class_name in enumerate(["Cat", "Dog"]):
            files = glob(str(Path(root_dir) / class_name / "*.npy"))
            self.samples.extend(files)
            self.labels.extend([label] * len(files))
        self.transform = self._build_transform()

    def _build_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                AddGaussianNoise(0., 0.05),  # Use custom class
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        arr = np.load(self.samples[idx])  # shape (224, 224, 3), float32, [0,1]
        arr = np.clip(arr, 0, 1)
        # Pass numpy array directly to transform pipeline
        img = self.transform(arr)
        label = self.labels[idx]
        return img, label

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total

def main():
    data_root = Path("data/ProcessedResizedNorm")
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameter grid
    learning_rates = [1e-3, 1e-4]
    batch_sizes = [16, 32]
    best_acc = 0.0
    best_params = None

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\n=== Training with lr={lr}, batch_size={batch_size} ===")
            train_ds = NPYImageDataset(data_root / "train", augment=True)
            val_ds = NPYImageDataset(data_root / "val", augment=False)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

            # Model: ResNet18
            model = models.resnet18(weights="IMAGENET1K_V1")
            model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            best_val_acc = 0.0
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
                print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model for this run
                    torch.save(model.state_dict(), f"best_cnn_model_lr{lr}_bs{batch_size}.pth")
            print(f"Best val acc for lr={lr}, batch_size={batch_size}: {best_val_acc:.4f}")
            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_params = {'learning_rate': lr, 'batch_size': batch_size}

    print(f"\nBest hyperparameters: {best_params}, Best validation accuracy: {best_acc:.4f}")

    # Optionally, evaluate on test set
    test_dir = data_root / "test"
    if test_dir.exists():
        test_ds = NPYImageDataset(test_dir, augment=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        model.load_state_dict(torch.load("best_cnn_model.pth"))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Evaluate model performance
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("Classification Report:")
    print(classification_report(y, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Check for over/underfitting
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.2%}")

    # Compare to baseline
    baseline_accuracy = 0.5  # Assuming a simple baseline
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    print(f"Improvement over baseline: {accuracy - baseline_accuracy:.2%}")

    # Measure of variance
    # Assuming PCA is applied to the CNN model, add the following lines
    # pca = pipeline.named_steps['pca']
    # explained_variance = np.sum(pca.explained_variance_ratio_)
    # print(f"Total explained variance for CNN: {explained_variance:.2%}")

if __name__ == "__main__":
    main() 