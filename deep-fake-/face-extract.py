import os
from pathlib import Path
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageDraw

def extract_faces_from_images(input_folder, output_folder, start_index=60000, end_index=65000, use_gpu=True):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Set GPU usage for face_recognition if available
    if use_gpu:
        face_recognition.api.batch_face_locations = face_recognition.api._raw_face_locations_batched
        face_recognition_model = "cnn"  # CNN model uses GPU if available
    else:
        face_recognition_model = "hog"  # HOG model is CPU-only
    
    # Get all supported image files from the directory
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.TIFF')
    all_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)])
    
    # Sanity check to make sure we have enough images
    if len(all_files) <= start_index:
        print(f"Warning: Only {len(all_files)} files found, but start_index is {start_index}")
        return
    
    if len(all_files) < end_index:
        print(f"Warning: Only {len(all_files)} files found, but end_index is {end_index}")
        end_index = len(all_files)
    
    # Process images in ascending order from start_index to end_index
    total_to_process = end_index - start_index
    processed_count = 0
    
    print(f"Starting to process {total_to_process} images from index {start_index} up to {end_index}")
    
    for i in range(start_index, end_index):
        filename = all_files[i]
        processed_count += 1
        
        try:
            # Load input image
            input_path = os.path.join(input_folder, filename)
            input_image = Image.open(input_path).convert("RGB")
            
            # Convert PIL image to numpy array for face detection
            input_np = np.array(input_image)
            
            # Detect faces using face_recognition with GPU model if enabled
            face_locations = face_recognition.face_locations(input_np, model=face_recognition_model)
            
            # Get the original file extension
            _, ext = os.path.splitext(filename)
            
            # Save the image with new naming format: img_<index>.<extension>
            new_filename = f"img_{i}{ext}"
            output_path = os.path.join(output_folder, new_filename)
            input_image.save(output_path)
            
            if not face_locations:
                print(f"[{i}/{end_index}] No face detected in img_{i}, but image saved anyway")
            else:
                print(f"[{i}/{end_index}] Saved image img_{i} with {len(face_locations)} faces detected")
                
        except Exception as e:
            print(f"[{i}/{end_index}] Error processing {filename}: {str(e)}")
        
        # Print progress every 100 images
        if processed_count % 100 == 0:
            print(f"Progress: {processed_count}/{total_to_process} images processed ({round(processed_count/total_to_process*100, 2)}%)")
    
    print(f"Processing complete! {processed_count} images processed.")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/merged_shuffled_images"
    output_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/extracted-faces-parallel-2"
    extract_faces_from_images(input_folder, output_folder, start_index=60000, end_index=65000, use_gpu=True)