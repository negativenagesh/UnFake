import os
import random
from PIL import Image
import shutil

# Path to the parent directory containing the 10 folders
parent_dir = "/home/vu-lab03-pc24/Downloads/deep-fake/image_folders"
# Output directory for merged and shuffled images
output_dir = "/home/vu-lab03-pc24/Downloads/deep-fake/merged_shuffled_images"

# Create output directory if it doesn’t exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to hold all image paths
all_images = []

# Iterate through each folder and collect image paths
for folder_name in os.listdir(parent_dir):
    folder_path = os.path.join(parent_dir, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Check if it’s an image file (add more extensions if needed)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                all_images.append(file_path)

# Shuffle the list of image paths
random.shuffle(all_images)

# Copy images to output directory with new names
for i, image_path in enumerate(all_images):
    img = Image.open(image_path)
    # New name like img_001.jpg, padded with zeros
    new_name = f"img_{str(i+1).zfill(3)}.{image_path.split('.')[-1]}"
    output_path = os.path.join(output_dir, new_name)
    shutil.copy(image_path, output_path)  # Copy instead of moving, to keep originals
    print(f"Copied: {new_name}")

print(f"Total images merged and shuffled: {len(all_images)}")