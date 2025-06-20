# Animal Binary Classifier - Applied ML

**Authors:**  
- Mohammed Usman Sarki – s5528178

## 1. Overview
We build a machine-learning pipeline to distinguish cats from dogs in images. Using a real-world 
Kaggle dataset of 25 000 JPG/PNG images (12 500 cats, 12 500 dogs), we'll clean, preprocess, train 
both a Random Forest baseline and a fine-tuned ResNet CNN, and evaluate with a suite of metrics to measure performance and robustness.

## 2. Preprocessing 

1. **Loading & Cleaning** (`remove.py`)  
   - Scan `PetImages/{Cat,Dog}` with PIL `verify()`, move unreadable files to `Corrupt/` and log them.  
2. **Train/Val/Test Split** (`split.py`)  
   - Stratified 80 % train / 10 % val / 10 % test with `random_state=42`.  
3. **Resizing & Normalization** (`resize.py`)  
   - Resize all images to 224×224 via bilinear interpolation.  
   - Convert to RGB and save as `.npy` arrays of `float32` in [0,1].  
4. **Augmentation** (during CNN training)  
   - Random flips, rotations, crops, color jitter, Gaussian noise.  
5. **Class-Imbalance Handling**  
   - Monitor post-cleaning counts; apply oversampling or class-weighted loss if needed.

## 3. Baseline Model (`RandomForest.py`)
- **Preprocessing:**  
  - Resize to 64×64, flatten to 1D vectors, normalize [0,1].  
  - Optional PCA (500 components) to reduce from ~12 288 dims → 500 dims.  
- **Model:**  
  - 100-tree Random Forest, 5-fold stratified CV.  
- **Results:**  
  | Fold | Accuracy | F1-score |  
  |:----:|:--------:|:--------:|  
  | 1    | 0.610    | 0.601    |  
  | 2    | 0.606    | 0.598    |  
  | 3    | 0.614    | 0.606    |  
  | 4    | 0.611    | 0.609    |  
  | 5    | 0.615    | 0.612    |  
  | **Mean ± Std** | **0.611 ± 0.003** | **0.605 ± 0.005** |  
- **Baseline Insight:** ≈61 % accuracy, consistent across folds; serves as a reference.

## 4. Proposed CNN Model (CNN Training Pipeline)
- **Preprocessing:**
  - Uses 224×224 RGB .npy arrays normalized to [0,1] (prepared via resize.py
  - Applies on-the-fly data augmentation during training
  - Random horizontal flip, rotation, resized crop, color jitter, and Gaussian noise
- **Architecture:**
  - Pretrained ResNet18, fine-tuned for binary classification (2 output classes)
  - Final layer replaced with nn.Linear(..., 2)
  - Trained with CrossEntropyLoss and Adam optimizer
- **Regularization:**
  - L2 weight decay
  - Data augmentation
- **Evaluation:**
  - Reported accuracy and F1-score on validation and (optional) test set
  - GradCAM used to visualize spatial attention of the CNN on input images
- **Output:**
  - Best model saved as models/best_cnn_model.pth

## API
- **Start server using uvicorn**
  - uvicorn main:app --reload
- **Open the interactive Swagger UI**
  - http://127.0.0.1:8000/docs
- **Usage**
  - Endpoint: POST / predict
  - Query parameters
    - model_type: "rf" (Random Forest) or "cnn" (Convolutional Neural Network)
    - include_visualization: true or false (default: true)
  - File input: Upload a JPG or PNG image
- **Responses**
  - label: Predicted class; "Cat" or "Dog"
  - probability: Model confidence (float)
  - variance: Uncertainty in prediction
  - visualization: Optional heatmap image (if enabled)

## Dataset can be found at: https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset?select=PetImages

## Installation
Install the required packages: 
pip install -r requirements.txt


