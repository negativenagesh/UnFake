import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from PIL import Image
import os
from pathlib import Path
import numpy as np
import functools

# Cache for models
_model_cache = {
    "text_to_image": None,
    "img_to_img": None
}

def load_models():
    """Load and cache the models to avoid redundant loading."""
    if _model_cache["text_to_image"] is None:
        model_id = "black-forest-labs/FLUX.1-schnell"
        pipe_t2i = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe_t2i = pipe_t2i.to("cuda")
        _model_cache["text_to_image"] = pipe_t2i
    
    if _model_cache["img_to_img"] is None:
        model_id = "black-forest-labs/FLUX.1-schnell"
        pipe_i2i = FluxImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe_i2i = pipe_i2i.to("cuda")
        _model_cache["img_to_img"] = pipe_i2i

def process_images_to_deepfake_flux(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load models (cached)
    load_models()
    pipe_t2i = _model_cache["text_to_image"]
    pipe_i2i = _model_cache["img_to_img"]
    
    # Define prompts
    face_prompt = (
        "A hyper-detailed, ultra-realistic face of a person, photorealistic, 8K resolution, "
        "exceptionally lifelike facial features with visible skin pores, subtle imperfections, and natural micro-textures, "
        "intricate eye details with realistic iris patterns, light reflections, and depth, "
        "individual hair strands with natural sheen and flow, soft cinematic lighting with delicate shadows and highlights, "
        "perfectly balanced color tones, professional studio photography quality, razor-sharp focus"
    )

    full_prompt = (
        "A hyper-detailed, ultra-realistic image of a person, photorealistic, 8K resolution, "
        "exceptionally lifelike facial features with visible skin pores, subtle imperfections, and natural micro-textures, "
        "intricate eye details with realistic iris patterns, light reflections, and depth, "
        "individual hair strands with natural sheen and flow, soft cinematic lighting with delicate shadows and highlights, "
        "perfectly balanced color tones, professional studio photography quality, razor-sharp focus"
    )

    negative_prompt = (
        "blurry, low resolution, pixelated, grainy, distorted, unrealistic, cartoonish, oversaturated, "
        "overexposed, underexposed, flat colors, dull, oversmoothed, overprocessed, synthetic look, plastic texture, "
        "uncanny valley, bad anatomy, extra limbs, mutated hands, deformed face, disfigured, poorly drawn eyes, "
        "poorly rendered skin, unnatural lighting, text, watermark, logo, signature, low quality, artifacts"
    )

    strength = 0.7
    guidance_scale = 12.0
    
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.TIFF')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            try:
                input_path = os.path.join(input_folder, filename)
                input_image = Image.open(input_path).convert("RGB")
                input_np = np.array(input_image)
                
                # Detect blacked-out region (assuming it's a pure black rectangle)
                black_mask = (input_np[:, :, 0] == 0) & (input_np[:, :, 1] == 0) & (input_np[:, :, 2] == 0)
                if not np.any(black_mask):
                    print(f"No blacked-out region detected in {filename}, proceeding with img2img directly")
                    processed_image = input_image
                else:
                    # Find the bounding box of the blacked-out region
                    coords = np.where(black_mask)
                    top, bottom = coords[0].min(), coords[0].max()
                    left, right = coords[1].min(), coords[1].max()
                    
                    # Add padding to the blacked-out region for better blending
                    padding_h = int((bottom - top) * 0.2)
                    padding_w = int((right - left) * 0.2)
                    face_top = max(0, top - padding_h)
                    face_bottom = min(input_np.shape[0], bottom + padding_h)
                    face_left = max(0, left - padding_w)
                    face_right = min(input_np.shape[1], right + padding_w)
                    
                    # Generate a face to replace the blacked-out region
                    with torch.autocast("cuda"):
                        generated_face = pipe_t2i(
                            prompt=face_prompt,
                            negative_prompt=negative_prompt,
                            height=512,
                            width=512,
                            guidance_scale=7.0
                        ).images[0]
                    
                    # Resize the generated face to match the blacked-out region
                    face_width = face_right - face_left
                    face_height = face_bottom - face_top
                    generated_face = generated_face.resize((face_width, face_height), Image.LANCZOS)
                    
                    # Blend the generated face into the input image
                    processed_image = input_image.copy()
                    processed_image.paste(generated_face, (face_left, face_top))
                
                # Resize the entire image for img2img processing
                processed_image = processed_image.resize((1024, 1024), Image.LANCZOS)
                
                # Apply img2img pipeline to enhance the entire image
                with torch.autocast("cuda"):
                    final_image = pipe_i2i(
                        prompt=full_prompt,
                        image=processed_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt
                    ).images[0]
                
                # Save the output image
                output_filename = f"deepfake_flux_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                final_image.save(output_path)
                
                print(f"Processed {filename} -> Saved as {output_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print("All images processed successfully!")

if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/facerec"
    output_folder = "/home/vu-lab03-pc24/Downloads/deep-fake/deep-fake-images-flux"
    process_images_to_deepfake_flux(input_folder, output_folder)