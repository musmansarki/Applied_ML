import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from pathlib import Path
import joblib
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import cv2
from gradcam import GradCAM
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="API for classifying images as cats or dogs using Random Forest and CNN models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class Prediction(BaseModel):
    label: str
    probability: float
    visualization: Optional[bytes] = None
    metrics: Optional[Dict[str, Any]] = None

# Define the ResNet model architecture
class ResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

# Global variables for loaded models
models_loaded = {
    'rf': None,
    'cnn': None
}

def load_models():
    """Load both models at startup"""
    models = {}
    try:
        
        rf_path = os.path.join('models', 'rf_pipeline.joblib')
        if os.path.exists(rf_path):
            models['rf'] = joblib.load(rf_path)
            print("✓ Random Forest model loaded")
        else:
            print("✗ Random Forest model file not found")
            
        
        cnn_path = os.path.join('models', 'best_cnn_model.pth')
        if os.path.exists(cnn_path):
            cnn_model = ResNet()
            state_dict = torch.load(cnn_path, map_location=torch.device('cpu'))
            # Handle the state dict mismatch by renaming keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'downsample' in k:
                    new_key = k.replace('downsample', 'shortcut')
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            cnn_model.load_state_dict(new_state_dict)
            cnn_model.eval()
            models['cnn'] = cnn_model
            print("✓ CNN model loaded")
        else:
            print("✗ CNN model file not found")
            
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        
    return models

# Load models at startup
@app.on_event("startup")
async def startup_event():
    models_loaded.update(load_models())

def preprocess_image_rf(image: Image.Image) -> np.ndarray:
    """Preprocess image for Random Forest model"""
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image to match training size
        image = image.resize((64, 64), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Ensure correct shape (64, 64, 3)
        if len(img_array.shape) == 2:  # If grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] != 3:  # If not RGB
            img_array = img_array[:, :, :3]  # Take first 3 channels
            
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Flatten the image
        return img_array.reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

def preprocess_image_cnn(image: Image.Image) -> torch.Tensor:
    """Preprocess image for CNN model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def visualize_cnn_prediction(img, probabilities):
    """Create visualization for CNN prediction using GradCAM and include variance measure"""
    # Convert numpy array to PIL Image if needed
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    
    # Get the last convolutional layer for GradCAM
    target_layer = models_loaded['cnn'].layer4[-1]
    
    # Initialize GradCAM
    grad_cam = GradCAM(models_loaded['cnn'], target_layer)
    
    # Preprocess image for model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    # Generate heatmap
    heatmap = grad_cam(input_tensor)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Blend heatmap with original image
    output = cv2.addWeighted(image_cv, 0.6, heatmap, 0.4, 0)
    
    # Convert back to PIL
    output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    # Save to bytes
    buf = io.BytesIO()
    output_image.save(buf, format='PNG')
    buf.seek(0)
    
    return buf.getvalue()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Cat vs Dog Classifier API is running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: str = Query(..., description="Model type: 'rf' or 'cnn'"),
    include_visualization: bool = Query(True, description="Include visualization in response")
):
    try:
        # Read and validate image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
            
        # Convert to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
            
        # Validate model type
        if model_type not in ['rf', 'cnn']:
            raise HTTPException(status_code=400, detail="Invalid model type. Use 'rf' or 'cnn'")
            
        # Check if model is loaded
        if model_type not in models_loaded:
            raise HTTPException(status_code=404, detail=f"{model_type.upper()} model not loaded")
            
        # Process image based on model type
        if model_type == 'rf':
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            # Preprocess image
            features = preprocess_image_rf(img_pil)
            print(f"[DEBUG] RF features shape: {features.shape}, dtype: {features.dtype}")
            import traceback
            try:
                print("[DEBUG] About to call RF predict...")
                prediction = models_loaded['rf'].predict(features)[0]
                print(f"[DEBUG] RF predict successful: {prediction}")
                print("[DEBUG] About to call RF predict_proba...")
                probabilities = models_loaded['rf'].predict_proba(features)[0]
                print(f"[DEBUG] RF predict_proba successful: {probabilities}")
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[ERROR] RF prediction failed with traceback:")
                print(tb)
                raise HTTPException(status_code=400, detail=f"RF prediction failed: {str(e)}\n\nTraceback:\n{tb}")
            
            # For RF: 0 is Cat, 1 is Dog
            predicted_class = "Cat" if prediction == 0 else "Dog"
            confidence = max(probabilities)
            variance = float(np.var(probabilities))  # Calculate variance from probabilities
            
            response_data = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "variance": variance,  # Add variance to response
                "probabilities": {
                    "Cat": float(probabilities[0]),
                    "Dog": float(probabilities[1])
                }
            }
            
            if include_visualization:
                try:
                    print("[DEBUG] Creating RF visualization...")
                    # Get feature importances from the Random Forest classifier
                    importances = models_loaded['rf'].named_steps['classifier'].feature_importances_
                    print(f"[DEBUG] Importances shape: {importances.shape}")
                    
                    # Create a bar plot of the top PCA component importances
                    plt.figure(figsize=(10, 6))
                    top_n = min(20, len(importances))  # Show top 20 components
                    top_indices = np.argsort(importances)[-top_n:]
                    top_importances = importances[top_indices]
                    
                    plt.barh(range(top_n), top_importances)
                    plt.yticks(range(top_n), [f'PC{i+1}' for i in top_indices])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top {top_n} PCA Component Importances (Random Forest)')
                    plt.gca().invert_yaxis()
                    
                    # Save the plot
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                    img_buffer.seek(0)
                    plt.close()
                    
                    # Convert to base64
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    response_data["visualization"] = f"data:image/png;base64,{img_base64}"
                    print("[DEBUG] RF visualization created successfully")
                except Exception as e:
                    tb = traceback.format_exc()
                    print(f"[ERROR] RF visualization failed: {str(e)}")
                    print(tb)
                    response_data["visualization_error"] = f"Failed to create visualization: {str(e)}"
            
            return response_data
        else:  # CNN
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            # Preprocess image
            img_tensor = preprocess_image_cnn(img_pil)
            # Get prediction
            with torch.no_grad():
                outputs = models_loaded['cnn'](img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                confidence = float(probabilities[prediction])
                variance = float(torch.var(probabilities).item())
            
            # For CNN: 0 is Cat, 1 is Dog
            predicted_class = "Cat" if prediction == 0 else "Dog"
            
            response_data = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "variance": variance,
                "probabilities": {
                    "Cat": float(probabilities[0]),
                    "Dog": float(probabilities[1])
                }
            }
            
            if include_visualization:
                try:
                    # Create the GradCAM visualization
                    visualization_bytes = visualize_cnn_prediction(img_rgb, probabilities.numpy())
                    
                    # Return the visualization as an image with prediction metadata in headers
                    return Response(
                        content=visualization_bytes,
                        media_type="image/png",
                        headers={
                            "Content-Disposition": f"inline; filename=gradcam_visualization.png",
                            "X-Predicted-Class": predicted_class,
                            "X-Confidence": str(confidence),
                            "X-Variance": str(variance),
                            "X-Probabilities": f"Cat:{probabilities[0]:.4f},Dog:{probabilities[1]:.4f}"
                        }
                    )
                except Exception as e:
                    response_data["visualization_error"] = f"Failed to create visualization: {str(e)}"
                    return response_data
            
            return response_data
        
    except Exception as e:
        print(f"Error details: {str(e)}")  # Debug print
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/evaluate")
async def evaluate_models():
    """Evaluate both models on test data and return metrics including variance"""
    try:
        # Load test data
        test_data = np.load('test_data.npy')
        test_labels = np.load('test_labels.npy')
        
        results = {}
        
        # Evaluate Random Forest
        if 'rf' in models_loaded:
            try:
                # Get predictions and probabilities
                rf_predictions = models_loaded['rf'].predict(test_data)
                rf_probabilities = models_loaded['rf'].predict_proba(test_data)
                
                # Calculate basic metrics
                rf_accuracy = accuracy_score(test_labels, rf_predictions)
                rf_precision = precision_score(test_labels, rf_predictions, average='weighted')
                rf_recall = recall_score(test_labels, rf_predictions, average='weighted')
                rf_f1 = f1_score(test_labels, rf_predictions, average='weighted')
                
                # Calculate variance metrics
                rf_variances = np.var(rf_probabilities, axis=1)
                rf_mean_variance = float(np.mean(rf_variances))
                rf_std_variance = float(np.std(rf_variances))
                rf_confidence_scores = np.max(rf_probabilities, axis=1)
                rf_mean_confidence = float(np.mean(rf_confidence_scores))
                
                results['random_forest'] = {
                    'accuracy': rf_accuracy,
                    'precision': rf_precision,
                    'recall': rf_recall,
                    'f1_score': rf_f1,
                    'mean_variance': rf_mean_variance,
                    'std_variance': rf_std_variance,
                    'mean_confidence': rf_mean_confidence,
                    'variance_distribution': {
                        'min': float(np.min(rf_variances)),
                        'max': float(np.max(rf_variances)),
                        'median': float(np.median(rf_variances))
                    }
                }
            except Exception as e:
                results['random_forest'] = {'error': str(e)}
        
        # Evaluate CNN
        if 'cnn' in models_loaded:
            try:
                # Convert test data to tensor format
                test_tensor = torch.FloatTensor(test_data)
                
                # Get predictions
                cnn_predictions = []
                cnn_probabilities = []
                cnn_variances = []
                
                with torch.no_grad():
                    for i in range(0, len(test_tensor), 32):  # Process in batches
                        batch = test_tensor[i:i+32]
                        outputs = models_loaded['cnn'](batch)
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        vars = torch.var(probs, dim=1)
                        
                        cnn_probabilities.extend(probs.numpy())
                        cnn_predictions.extend(preds.numpy())
                        cnn_variances.extend(vars.numpy())
                
                cnn_predictions = np.array(cnn_predictions)
                cnn_probabilities = np.array(cnn_probabilities)
                cnn_variances = np.array(cnn_variances)
                
                # Calculate basic metrics
                cnn_accuracy = accuracy_score(test_labels, cnn_predictions)
                cnn_precision = precision_score(test_labels, cnn_predictions, average='weighted')
                cnn_recall = recall_score(test_labels, cnn_predictions, average='weighted')
                cnn_f1 = f1_score(test_labels, cnn_predictions, average='weighted')
                
                # Calculate variance metrics
                cnn_mean_variance = float(np.mean(cnn_variances))
                cnn_std_variance = float(np.std(cnn_variances))
                cnn_confidence_scores = np.max(cnn_probabilities, axis=1)
                cnn_mean_confidence = float(np.mean(cnn_confidence_scores))
                
                results['cnn'] = {
                    'accuracy': cnn_accuracy,
                    'precision': cnn_precision,
                    'recall': cnn_recall,
                    'f1_score': cnn_f1,
                    'mean_variance': cnn_mean_variance,
                    'std_variance': cnn_std_variance,
                    'mean_confidence': cnn_mean_confidence,
                    'variance_distribution': {
                        'min': float(np.min(cnn_variances)),
                        'max': float(np.max(cnn_variances)),
                        'median': float(np.median(cnn_variances))
                    }
                }
            except Exception as e:
                results['cnn'] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

