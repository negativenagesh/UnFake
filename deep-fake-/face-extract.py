import os
from pathlib import Path
import numpy as np
import cv2
import face_recognition
from PIL import Image

def extract_faces_from_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
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
                
                # Detect faces using face_recognition
                face_locations = face_recognition.face_locations(input_np)
                
                if not face_locations:
                    print(f"No face detected in {filename}, skipping")
                    continue
                
                # Process each detected face
                for i, face_location in enumerate(face_locations):
                    top, right, bottom, left = face_location
                    
                    # Add padding to face for better results (30% padding to include more of head/neck)
                    height, width = bottom - top, right - left
                    padding_h = int(height * 0.3)
                    padding_w = int(width * 0.3)
                    
                    # Apply padding with boundary checks
                    # Add more padding at the bottom for neck
                    face_top = max(0, top - padding_h)
                    face_bottom = min(input_np.shape[0], bottom + padding_h * 2)  # Extra padding for neck
                    face_left = max(0, left - padding_w)
                    face_right = min(input_np.shape[1], right + padding_w)
                    
                    # Extract the face region
                    face_img = input_image.crop((face_left, face_top, face_right, face_bottom))
                    
                    # Save the extracted face
                    face_filename = f"face_{i+1}_{filename}"
                    output_path = os.path.join(output_folder, face_filename)
                    face_img.save(output_path)
                    
                    print(f"Extracted face {i+1} from {filename} -> Saved as {face_filename}")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print("All faces extracted successfully!")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/merged_shuffled_images"
    output_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/extracted-faces"
    extract_faces_from_images(input_folder, output_folder)