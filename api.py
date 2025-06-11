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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import cv2
from gradcam import GradCAM

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
        # Load Random Forest model
        rf_path = os.path.join('models', 'rf_pipeline.joblib')
        if os.path.exists(rf_path):
            models['rf'] = joblib.load(rf_path)
            print("✓ Random Forest model loaded")
        else:
            print("✗ Random Forest model file not found")
            
        # Load CNN model
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
    # Resize image
    image = image.resize((64, 64))
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    # Flatten the image
    return img_array.reshape(1, -1)

def preprocess_image_cnn(image: Image.Image) -> torch.Tensor:
    """Preprocess image for CNN model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def visualize_rf_prediction(image, importances):
    """Create visualization for Random Forest prediction using GradCAM-like approach."""
    # Convert image to numpy array if it's not already
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Reshape importances to match PCA components (10x10 grid)
    importance_map = importances.reshape(10, 10)
    
    # Resize importance map to match original image size
    importance_map = cv2.resize(importance_map, (image.shape[1], image.shape[0]))
    
    # Normalize importance map to [0, 1]
    importance_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())
    
    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * importance_map), cv2.COLORMAP_JET)
    
    # Convert image to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Blend heatmap with original image
    output = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0)
    
    # Convert back to RGB
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    output_image = Image.fromarray(output_rgb)
    
    # Save to bytes
    buf = io.BytesIO()
    output_image.save(buf, format='PNG')
    buf.seek(0)
    
    return buf.getvalue()

def visualize_cnn_prediction(img, probabilities):
    """Create visualization for CNN prediction using GradCAM"""
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
    
    # Convert back to PIL Image
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
            # Get prediction
            prediction = models_loaded['rf'].predict(features)[0]
            probabilities = models_loaded['rf'].predict_proba(features)[0]
            confidence = float(probabilities[1] if prediction == 1 else probabilities[0])
            
            # Create visualization if requested
            visualization = None
            if include_visualization:
                # Get feature importances from the classifier step
                importances = models_loaded['rf'].named_steps['classifier'].feature_importances_
                # Create visualization
                visualization = visualize_rf_prediction(img_rgb, importances)
                
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
            
            # Create visualization if requested
            visualization = None
            if include_visualization:
                visualization = visualize_cnn_prediction(img_rgb, probabilities.numpy())
        
        metrics = {
            "confidence": confidence,
            "prediction_time": 0.0  # Add timing if needed
        }
        
        if include_visualization and visualization:
            return Response(
                content=visualization,
                media_type="image/png",
                headers={
                    "X-Prediction-Label": "Cat" if prediction == 1 else "Dog",
                    "X-Prediction-Confidence": str(confidence)
                }
            )
        else:
            return JSONResponse({
                "label": "Cat" if prediction == 1 else "Dog",
                "probability": confidence,
                "metrics": metrics
            })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

