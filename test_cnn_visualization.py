#!/usr/bin/env python3
"""
Test script to verify CNN visualization is working correctly
"""

import requests
import base64
import io
from PIL import Image

def test_cnn_visualization():
    """Test the CNN visualization endpoint"""
    
    # Test image (you can replace this with any cat/dog image)
    test_image_path = "test_image.jpg"  # Replace with your test image
    
    try:
        # Make request to CNN endpoint
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                'http://localhost:8000/predict?model_type=cnn&include_visualization=true',
                files=files
            )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ CNN prediction successful!")
            print(f"Predicted class: {data['predicted_class']}")
            print(f"Confidence: {data['confidence']:.4f}")
            print(f"Variance: {data['variance']:.4f}")
            
            if 'visualization' in data:
                print("✅ Visualization found!")
                visualization_data = data['visualization']
                
                # Extract base64 data
                if visualization_data.startswith('data:image/png;base64,'):
                    base64_data = visualization_data.split(',')[1]
                    
                    # Decode and save the image
                    image_data = base64.b64decode(base64_data)
                    with open('cnn_visualization.png', 'wb') as f:
                        f.write(image_data)
                    
                    print("✅ Visualization saved as 'cnn_visualization.png'")
                    print("You can now open this file to see the GradCAM heatmap!")
                    
                    # Also display the image if PIL is available
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        print(f"Image size: {img.size}")
                        print("Image format: PNG")
                    except Exception as e:
                        print(f"Could not display image: {e}")
                else:
                    print("❌ Invalid visualization format")
            else:
                print("❌ No visualization in response")
                if 'visualization_error' in data:
                    print(f"Visualization error: {data['visualization_error']}")
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Error: {response.text}")
            
    except FileNotFoundError:
        print(f"❌ Test image not found: {test_image_path}")
        print("Please provide a valid test image path")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cnn_visualization() 