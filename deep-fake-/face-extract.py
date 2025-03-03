import os
from pathlib import Path
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageDraw

def extract_faces_from_images(input_folder, output_folder, use_gpu=True):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Set GPU usage for face_recognition if available
    if use_gpu:
        face_recognition.api.batch_face_locations = face_recognition.api._raw_face_locations_batched
        face_recognition_model = "cnn"  # CNN model uses GPU if available
    else:
        face_recognition_model = "hog"  # HOG model is CPU-only
    
    # Process each image in the input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.TIFF')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            try:
                # Load input image
                input_path = os.path.join(input_folder, filename)
                input_image = Image.open(input_path).convert("RGB")
                
                # Convert PIL image to numpy array for face detection
                input_np = np.array(input_image)
                
                # Detect faces using face_recognition with GPU model if enabled
                face_locations = face_recognition.face_locations(input_np, model=face_recognition_model)
                
                # Save the image regardless of face detection
                output_path = os.path.join(output_folder, filename)
                input_image.save(output_path)
                
                if not face_locations:
                    print(f"No face detected in {filename}, but image saved anyway")
                else:
                    print(f"Saved image {filename} with {len(face_locations)} faces detected")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print("All images saved successfully!")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/merged_shuffled_images"
    output_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/extracted-faces"
    extract_faces_from_images(input_folder, output_folder, use_gpu=True)
