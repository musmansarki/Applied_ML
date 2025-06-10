from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from pathlib import Path
from PIL import Image
import numpy as np
import io, joblib
import torch
from torchvision import models, transforms
from gradcam import get_gradcam_for_prediction
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, explained_variance_score
from typing import Optional, List, Dict, Any
import base64

# Model paths
MODEL_DIR = Path(__file__).resolve().parent / 'models'
RF_MODEL_PATH = MODEL_DIR / 'rf_pipeline.joblib'
CNN_MODEL_PATH = MODEL_DIR / 'best_cnn_model.pth'

# Load models
models_loaded = {}
try:
    models_loaded['rf'] = joblib.load(RF_MODEL_PATH)
    print("✓ Random Forest model loaded")
except Exception as e:
    print(f"✗ Error loading Random Forest model: {e}")

try:
    models_loaded['cnn'] = torch.load(CNN_MODEL_PATH)
    models_loaded['cnn'].eval()
    print("✓ CNN model loaded")
except Exception as e:
    print(f"✗ Error loading CNN model: {e}")

if not models_loaded:
    print("Warning: No models found. Please train models first.")

# FastAPI setup
app = FastAPI(
    title="Cat vs Dog Classifier API",
    version="1.0",
    description="""
    A FastAPI-based API for classifying images as either 'Cat' or 'Dog'. 
    Supports both Random Forest and CNN models.
    
    ## Features
    * Image classification using Random Forest or CNN models
    * Support for both model types
    * Real-time predictions
    * Interactive API documentation
    
    ## Models
    * Random Forest: Traditional machine learning approach
    * CNN: Deep learning approach using ResNet18 architecture
    
    ## Usage
    1. Check available models at `/models`
    2. Upload an image to `/predict` with your chosen model
    3. Get prediction results with confidence scores
    """,
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and system information endpoints",
            "externalDocs": {
                "description": "API Documentation",
                "url": "http://127.0.0.1:8000/docs",
            },
        },
        {
            "name": "predict",
            "description": "Image classification endpoints",
            "externalDocs": {
                "description": "Example Usage",
                "url": "http://127.0.0.1:8000/docs#/predict/predict_predict_post",
            },
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

class Prediction(BaseModel):
    label: str = Field(..., description="Predicted class (Cat or Dog)")
    probability: float = Field(..., description="Confidence score for the prediction")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Additional metrics for this prediction")

    class Config:
        schema_extra = {
            "example": {
                "label": "Cat",
                "probability": 0.95,
                "visualization": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAACXBIWXMAAAsTAAALEwEAmpwYAAAKT2lDQ1BQaG90b3Nob3AgSUN... (base64 encoded image data)",
                "metrics": {
                    "feature_importance": [0.1, 0.2, 0.3, 0.4],
                    "class_probabilities": [0.9, 0.1],
                }
            }
        }

class ModelInfo(BaseModel):
    available_models: list[str] = Field(..., description="List of available model types")

    class Config:
        schema_extra = {
            "example": {
                "available_models": ["rf", "cnn"]
            }
        }

class EvaluationMetrics(BaseModel):
    accuracy: float
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    feature_importance: Optional[List[float]] = None

@app.get("/", tags=["health"], response_description="API health status and documentation links")
def health_check():
    """
    Health check endpoint to verify the API is running.
    Returns a simple status message, available models, and links to documentation.
    """
    base_url = "http://127.0.0.1:8000"
    return {
        "status": "running",
        "available_models": list(models_loaded.keys()),
        "documentation": {
            "swagger_ui": f"{base_url}/docs",
            "redoc": f"{base_url}/redoc",
            "openapi_schema": f"{base_url}/openapi.json"
        },
        "endpoints": {
            "health_check": f"{base_url}/",
            "models_info": f"{base_url}/models",
            "predict": f"{base_url}/predict"
        },
        "api_spec": {
            "openapi": "3.0.0",
            "info": {
                "title": "Cat vs Dog Classifier API",
                "version": "1.0",
                "description": "API for classifying images as either 'Cat' or 'Dog'"
            },
            "servers": [
                {
                    "url": base_url,
                    "description": "Local development server"
                }
            ]
        }
    }

@app.get("/models", response_model=ModelInfo, tags=["health"], response_description="List of available models")
def get_models():
    """
    Get information about available models.
    Returns a list of model names that are currently loaded.
    """
    return ModelInfo(available_models=list(models_loaded.keys()))

def preprocess_image(img: Image.Image, model_type: str = 'rf'):
    """Preprocess image for model prediction"""
    if model_type == 'rf':
        # Resize to 64×64 for RF model (to match training)
        img = img.resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.flatten().reshape(1, -1)
    else:  # CNN
        # Use the same transforms as in training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        return transform(img).unsqueeze(0)

def visualize_rf_prediction(img: Image.Image, feature_importance: np.ndarray) -> str:
    """Generate heatmap visualization for Random Forest prediction."""
    # Reshape feature importance to match image dimensions
    importance_2d = feature_importance.reshape(64, 64)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(importance_2d, cmap='hot')
    plt.title('Feature Importance Heatmap')
    plt.axis('off')
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    
    # Convert to base64
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

def generate_gradcam(model: torch.nn.Module, img_tensor: torch.Tensor, pred_label: int) -> str:
    """Generate GradCAM visualization for CNN prediction."""
    # Implementation of GradCAM
    # ... (keep existing GradCAM implementation)
    pass

@app.post("/predict", response_model=Prediction, tags=["predict"])
async def predict(
    file: UploadFile = File(..., description="Image file to classify (JPEG, PNG)"),
    model_type: str = "rf",
    include_visualization: bool = False
):
    """Make a prediction for a single image with optional visualization."""
    if model_type not in models_loaded:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        if model_type == 'rf':
            # Preprocess for RF
            img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)
            features = np.array(img_resized).flatten() / 255.0
            
            # Get prediction
            pred_probs = models_loaded['rf'].predict_proba([features])[0]
            pred_label = int(np.argmax(pred_probs))
            confidence = float(np.max(pred_probs))
            
            # Get feature importance
            feature_importance = models_loaded['rf'].named_steps['rf'].feature_importances_
            
            # Generate visualization if requested
            visualization = None
            if include_visualization:
                visualization = visualize_rf_prediction(img_resized, feature_importance)
            
            return {
                "label": ["Cat", "Dog"][pred_label],
                "probability": confidence,
                "visualization": visualization,
                "metrics": {
                    "feature_importance": feature_importance.tolist(),
                    "class_probabilities": pred_probs.tolist()
                }
            }
            
        else:  # CNN
            # Preprocess for CNN
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            img_tensor = transform(img).unsqueeze(0)
            
            # Get prediction
            model = models_loaded['cnn']
            model.eval()
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_label = int(torch.argmax(probs))
                confidence = float(probs[pred_label])
            
            # Generate visualization if requested
            visualization = None
            if include_visualization:
                visualization = generate_gradcam(model, img_tensor, pred_label)
            
            return {
                "label": ["Cat", "Dog"][pred_label],
                "probability": confidence,
                "visualization": visualization,
                "metrics": {
                    "class_probabilities": probs.cpu().numpy().tolist()
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/evaluate_rf", response_model=EvaluationMetrics, tags=["evaluate"])
async def evaluate_rf():
    """Get overall evaluation metrics for Random Forest model on test set."""
    if 'rf' not in models_loaded:
        raise HTTPException(status_code=404, detail="Random Forest model not found")
    
    test_path = Path(__file__).resolve().parent / 'data' / 'ProcessedResizedNorm' / 'test'
    X_test, y_test = load_test_data(test_path)
    
    # Make predictions
    y_pred = models_loaded['rf'].predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "feature_importance": models_loaded['rf'].named_steps['rf'].feature_importances_.tolist()
    }

@app.get("/evaluate_cnn", response_model=EvaluationMetrics, tags=["evaluate"])
async def evaluate_cnn():
    """Get overall evaluation metrics for CNN model on test set."""
    if 'cnn' not in models_loaded:
        raise HTTPException(status_code=404, detail="CNN model not found")
    
    test_path = Path(__file__).resolve().parent / 'data' / 'ProcessedResizedNorm' / 'test'
    X_test, y_test = load_cnn_test_data(test_path)
    
    model = models_loaded['cnn']
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_test)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, output_dict=True)
    conf_matrix = confusion_matrix(y_true, preds).tolist()
    
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }

@app.get("/evaluate_per_image", response_model=list, tags=["predict"], response_description="Per-image evaluation for Random Forest")
async def evaluate_per_image():
    if 'rf' not in models_loaded:
        raise HTTPException(status_code=404, detail="Random Forest model not found")
    test_path = Path(__file__).resolve().parent / 'data' / 'ProcessedResizedNorm' / 'test'
    results = []
    for label, class_name in enumerate(['Cat', 'Dog']):
        class_dir = test_path / class_name
        if not class_dir.exists():
            continue
        for file in class_dir.glob('*.npy'):
            arr = np.load(file)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            feats = np.array(img).flatten() / 255.0
            pred_probs = models_loaded['rf'].predict_proba([feats])[0]
            pred_label = int(np.argmax(pred_probs))
            confidence = float(np.max(pred_probs))
            results.append({
                "filename": str(file.name),
                "true_label": class_name,
                "predicted_label": ["Cat", "Dog"][pred_label],
                "confidence": confidence
            })
    return results

@app.get("/evaluate_cnn_per_image", response_model=list, tags=["predict"], response_description="Per-image evaluation for CNN")
async def evaluate_cnn_per_image():
    if 'cnn' not in models_loaded:
        raise HTTPException(status_code=404, detail="CNN model not found")
    test_path = Path(__file__).resolve().parent / 'data' / 'ProcessedResizedNorm' / 'test'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    model = models_loaded['cnn']
    model.eval()
    results = []
    for label, class_name in enumerate(['Cat', 'Dog']):
        class_dir = test_path / class_name
        if not class_dir.exists():
            continue
        for file in class_dir.glob('*.npy'):
            arr = np.load(file)
            arr = np.clip(arr, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_label = int(np.argmax(probs))
                confidence = float(np.max(probs))
            results.append({
                "filename": str(file.name),
                "true_label": class_name,
                "predicted_label": ["Cat", "Dog"][pred_label],
                "confidence": confidence
            })
    return results

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

def load_test_data(test_path):
    X, y = [], []
    for label, class_name in enumerate(['Cat', 'Dog']):
        class_dir = test_path / class_name
        if not class_dir.exists():
            continue
        for file in class_dir.glob('*.npy'):
            arr = np.load(file)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.resize((64, 64), Image.Resampling.LANCZOS)
            X.append(np.array(img).flatten() / 255.0)
            y.append(label)
    return np.array(X), np.array(y)

def load_cnn_test_data(test_path):
    X, y = [], []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    for label, class_name in enumerate(['Cat', 'Dog']):
        class_dir = test_path / class_name
        if not class_dir.exists():
            continue
        for file in class_dir.glob('*.npy'):
            arr = np.load(file)
            arr = np.clip(arr, 0, 1)
            img = Image.fromarray((arr * 255).astype(np.uint8))
            img = img.convert('RGB')
            img = transform(img)
            X.append(img)
            y.append(label)
    X = torch.stack(X)
    y = torch.tensor(y)
    return X, y

