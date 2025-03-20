import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import os
from pathlib import Path
import numpy as np
import face_recognition

def process_images_to_deepfake_sdxl(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    prompt = "A hyper-detailed, ultra-realistic portrait of the person's face, photorealistic, 8K resolution, exceptionally lifelike facial features with visible skin pores, subtle imperfections, and natural micro-textures, intricate eye details with realistic iris patterns, light reflections, and depth, individual hair strands with natural sheen and flow, soft cinematic lighting with delicate shadows and highlights, perfectly balanced color tones, professional studio photography quality, razor-sharp focus"
    negative_prompt = "blurry, low resolution, pixelated, grainy, distorted, unrealistic, cartoonish, oversaturated, overexposed, underexposed, flat colors, dull, oversmoothed, overprocessed, synthetic look, plastic texture, uncanny valley, bad anatomy, extra limbs, mutated hands, deformed face, disfigured, poorly drawn eyes, poorly rendered skin, unnatural lighting, text, watermark, logo, signature, low quality, artifacts"
    strength = 0.85
    guidance_scale = 12.0
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.TIFF')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            try:
                input_path = os.path.join(input_folder, filename)
                input_image = Image.open(input_path).convert("RGB")
                input_np = np.array(input_image)
                face_locations = face_recognition.face_locations(input_np)
                
                if not face_locations:
                    print(f"No face detected in {filename}, skipping")
                    continue
                
                for face_location in face_locations:
                    top, right, bottom, left = face_location
                    height, width = bottom - top, right - left
                    padding_h, padding_w = int(height * 0.2), int(width * 0.2)
                    face_top = max(0, top - padding_h)
                    face_bottom = min(input_np.shape[0], bottom + padding_h)
                    face_left = max(0, left - padding_w)
                    face_right = min(input_np.shape[1], right + padding_w)
                    
                    face_img = input_image.crop((face_left, face_top, face_right, face_bottom))
                    face_img = face_img.resize((1024, 1024), Image.LANCZOS)
                    
                    with torch.autocast("cuda"):
                        deepfaked_face = pipe(prompt=prompt, image=face_img, strength=strength, guidance_scale=guidance_scale, negative_prompt=negative_prompt).images[0]
                    
                    deepfaked_face = deepfaked_face.resize((face_right - face_left, face_bottom - face_top), Image.LANCZOS)
                    result_image = input_image.copy()
                    result_image.paste(deepfaked_face, (face_left, face_top))
                    
                    output_filename = f"deepfake_sdxl_{filename}"
                    output_path = os.path.join(output_folder, output_filename)
                    result_image.save(output_path)
                    print(f"Processed face in {filename} -> Saved as {output_filename}")
                    break
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print("All images processed successfully!")

if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/facerec"
    output_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/deep-fake-images-sdxl"
    process_images_to_deepfake_sdxl(input_folder, output_folder)