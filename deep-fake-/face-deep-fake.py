import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from pathlib import Path
import numpy as np
import cv2
def process_images_to_deepfake(input_folder, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load the pre-trained Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Move the model to GPU
    
    # Define the prompt and parameters
    prompt = (
        "A hyper-detailed, ultra-realistic portrait of the person's face, photorealistic, 8K resolution, "
        "exceptionally lifelike facial features with visible skin pores, subtle imperfections, and natural micro-textures, "
        "intricate eye details with realistic iris patterns, light reflections, and depth, individual hair strands with natural sheen and flow, "
        "soft cinematic lighting with delicate shadows and highlights, perfectly balanced color tones, "
        "professional studio photography quality, razor-sharp focus"
    )

    negative_prompt = (
        "blurry, low resolution, pixelated, grainy, distorted, unrealistic, cartoonish, oversaturated, "
        "overexposed, underexposed, flat colors, dull, oversmoothed, overprocessed, synthetic look, plastic texture, "
        "uncanny valley, bad anatomy, extra limbs, mutated hands, deformed face, disfigured, poorly drawn eyes, "
        "poorly rendered skin, unnatural lighting, text, watermark, logo, signature, low quality, artifacts"
    )

    strength = 0.8  # Controls the influence of the input image on the output
    guidance_scale = 15.0  # How closely to follow the prompt
    
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
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    
                    # Add padding to face for better results (20% padding)
                    height, width = bottom - top, right - left
                    padding_h = int(height * 0.2)
                    padding_w = int(width * 0.2)
                    
                    # Apply padding with boundary checks
                    face_top = max(0, top - padding_h)
                    face_bottom = min(input_np.shape[0], bottom + padding_h)
                    face_left = max(0, left - padding_w)
                    face_right = min(input_np.shape[1], right + padding_w)
                    
                    # Extract and process the face region
                    face_img = input_image.crop((face_left, face_top, face_right, face_bottom))
                    
                    # Ensure face image is reasonable size for the model (512x512 works well with SD)
                    face_img = face_img.resize((512, 512), Image.LANCZOS)
                    
                    # Generate the deepfake face
                    with torch.autocast("cuda"):
                        deepfaked_face = pipe(
                            prompt=prompt,
                            image=face_img,
                            strength=strength,
                            guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt
                        ).images[0]
                    
                    # Resize back to original face dimensions
                    deepfaked_face = deepfaked_face.resize((face_right - face_left, face_bottom - face_top), Image.LANCZOS)
                    
                    # Create a copy of original image and paste the deepfaked face
                    result_image = input_image.copy()
                    result_image.paste(deepfaked_face, (face_left, face_top))
                    
                    # Save the output image
                    output_filename = f"deepfake_{filename}"
                    output_path = os.path.join(output_folder, output_filename)
                    result_image.save(output_path)
                    
                    print(f"Processed face in {filename} -> Saved as {output_filename}")
                    
                    # Only process the first face for simplicity
                    # Remove this break if you want to process all faces in an image
                    break
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print("All images processed successfully!")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/UnFake/images/oldman-face.jpeg"
    output_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/"
    process_images_to_deepfake(input_folder, output_folder)