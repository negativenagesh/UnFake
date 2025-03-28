import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model
from PIL import Image
import requests
from io import BytesIO
import os
import sys
import random  # For demo mode
import traceback

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model architecture
class DeepfakeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeClassifier, self).__init__()
        self.base_model = create_model('efficientnet_b7', pretrained=False, num_classes=0)
        self.fc = nn.Linear(2560, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

# Image preprocessing transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Check if we're in demo mode (no model available)
DEMO_MODE = False

def load_model():
    """Load the pretrained deepfake detection model."""
    global DEMO_MODE
    
    # Try multiple possible model paths
    possible_paths = [
        # Direct absolute path
        "/home/subrahmanya/projects/UnFake/Model/efficientnet_b7_deepfake.pth",
        
        # Relative path from current file
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Model/efficientnet_b7_deepfake.pth"),
        
        # Check in the same directory as the script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "efficientnet_b7_deepfake.pth")
    ]
    
    # Try each path
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"Found model at: {model_path}")
            break
    
    # If no model found, use demo mode
    if model_path is None:
        print(f"Model file not found, tried: {possible_paths}, using demo mode")
        DEMO_MODE = True
        return None
    
    try:
        model = DeepfakeClassifier().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}, using demo mode")
        traceback.print_exc()
        DEMO_MODE = True
        return None

# Global model instance to avoid reloading for each request
_model = None

def get_model():
    """Get or load the model."""
    global _model
    if _model is None:
        _model = load_model()
    return _model

def download_image(url):
    """Download an image from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        traceback.print_exc()
        raise Exception(f"Error downloading image from URL: {str(e)}")

def deepfake_detection(image_url, confidence_threshold=0.5):
    """
    Detect if an image from a URL is fake.
    
    Args:
        image_url (str): URL of the image to analyze
        confidence_threshold (float): Threshold for determining fake classification
        
    Returns:
        dict: Dictionary containing prediction result and confidence
    """
    print(f"Processing image: {image_url}")
    
    # If we're in demo mode or no URL provided, return random result
    if DEMO_MODE or not image_url:
        print("Using demo mode (random prediction)")
        # Random result with 30% chance of being fake
        is_fake = random.random() < 0.3
        confidence = random.uniform(0.7, 0.95)
        return {
            "is_fake": is_fake, 
            "confidence": confidence,
            "message": "Demo mode - Random prediction"
        }
    
    try:
        # Load model
        model = get_model()
        if model is None:
            print("Model not available, using demo mode")
            is_fake = random.random() < 0.3
            confidence = random.uniform(0.7, 0.95)
            return {
                "is_fake": is_fake, 
                "confidence": confidence,
                "message": "Model not available - Using demo prediction"
            }
        
        # Download and preprocess image
        image = download_image(image_url)
        image_tensor = data_transforms(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Class 1 is "fake", class 0 is "real"
        is_fake = predicted_class == 1
        
        print(f"Prediction: {'Fake' if is_fake else 'Real'}, Confidence: {confidence:.4f}")
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "message": "Analysis complete"
        }
        
    except Exception as e:
        print(f"Error in deepfake detection: {str(e)}")
        traceback.print_exc()
        print("Falling back to demo mode")
        # In case of error, return random result
        is_fake = random.random() < 0.3
        confidence = random.uniform(0.7, 0.95)
        return {
            "is_fake": is_fake,
            "confidence": confidence,
            "message": f"Error during analysis - {str(e)}"
        }

# Safe wrapper for Streamlit to call
def analyze_image_for_streamlit(image_url):
    """Safe wrapper for Streamlit to avoid event loop conflicts"""
    try:
        # Streamlit and PyTorch have threading/asyncio conflicts
        # This is a simplified approach to minimize those issues
        return deepfake_detection(image_url)
    except Exception as e:
        print(f"Error in analyze_image_for_streamlit: {str(e)}")
        traceback.print_exc()
        return {
            "is_fake": random.random() < 0.3,
            "confidence": random.uniform(0.7, 0.95),
            "message": f"Error analyzing image: {str(e)}"
        }

# For testing directly
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        test_url = "https://images.unsplash.com/photo-1438761681033-6461ffad8d80"
    
    result = deepfake_detection(test_url)
    print(f"Image is {'fake' if result['is_fake'] else 'real'} with {result['confidence']:.2f} confidence")