import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from torchvision import models, transforms
from pathlib import Path

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, index=None):
        # Forward pass
        self.model.eval()
        output = self.model(x)
        
        if index is None:
            index = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Weighted combination of feature maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # ReLU on the weighted combination
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.numpy()

def visualize_gradcam(model, image_path, target_layer, output_path=None):
    """
    Generate and visualize GradCAM for a given image.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        target_layer: Target layer for GradCAM
        output_path: Path to save visualization (optional)
    
    Returns:
        PIL Image with GradCAM overlay
    """
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Initialize GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = grad_cam(input_tensor)
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Blend heatmap with original image
    output = cv2.addWeighted(image_cv, 0.6, heatmap, 0.4, 0)
    
    # Convert back to PIL Image
    output_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    
    if output_path:
        output_image.save(output_path)
    
    return output_image

def get_gradcam_for_prediction(model, image_path, output_path=None):
    """
    Generate GradCAM visualization for a model prediction.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        output_path: Path to save visualization (optional)
    
    Returns:
        tuple: (prediction label, confidence, visualization image)
    """
    # Get the last convolutional layer (for ResNet18)
    target_layer = model.layer4[-1]
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[pred_idx].item()
        label = "Cat" if pred_idx == 0 else "Dog"
    
    # Generate GradCAM visualization
    visualization = visualize_gradcam(model, image_path, target_layer, output_path)
    
    return label, confidence, visualization 