import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from pathlib import Path

def process_images_to_deepfake(input_folder, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Load the pre-trained Stable Diffusion model
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Move the model to GPU
    
    # Define the prompt and parameters
    prompt = (
        "A hyper-detailed, ultra-realistic portrait of the person from the input image, photorealistic, 8K resolution, "
        "exceptionally lifelike facial features with visible skin pores, subtle imperfections, and natural micro-textures, "
        "intricate eye details with realistic iris patterns, light reflections, and depth, individual hair strands with natural sheen and flow, "
        "soft cinematic lighting with delicate shadows and highlights, perfectly balanced color tones, "
        "professional studio photography quality, razor-sharp focus, micro-details like faint freckles or fine wrinkles, "
        "dynamic depth of field, rendered in the style of Unreal Engine 5 hyper-realism, trending on ArtStation, "
        "no digital noise or artifacts, maximum fidelity, lifelike presence and emotional authenticity"
    )

    negative_prompt = (
        "blurry, low resolution, pixelated, grainy, distorted, unrealistic, cartoonish, oversaturated, "
        "overexposed, underexposed, flat colors, dull, oversmoothed, overprocessed, synthetic look, plastic texture, "
        "uncanny valley, bad anatomy, extra limbs, mutated hands, deformed face, disfigured, poorly drawn eyes, "
        "poorly rendered skin, unnatural lighting, text, watermark, logo, signature, low quality, artifacts, amateurish"
    )

    strength = 1.0  # Controls the influence of the input image on the output
    guidance_scale = 17.0  # Increased for stricter prompt adherence
    
    # Process each image in the input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.heic', '.raw', '.svg','.JPG', '.JPEG', '.PNG', '.BMP', '.WEBP', '.TIFF', '.HEIC', '.RAW', '.SVG')
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            try:
                # Load and prepare input image
                input_path = os.path.join(input_folder, filename)
                input_image = Image.open(input_path).convert("RGB")
                width, height = input_image.size
                input_image = input_image.resize((width, height))
                
                # Generate the deep fake image
                with torch.autocast("cuda"):
                    output_image = pipe(
                        prompt=prompt,
                        image=input_image,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt
                    ).images[0]
                
                # Save the output image
                output_filename = f"deepfake_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                output_image.save(output_path)
                
                print(f"Processed {filename} -> Saved as {output_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    
    print("All images processed successfully!")

# Example usage
if __name__ == "__main__":
    input_folder = "/home/vu-lab03-pc24/Downloads/image_folders/closeup-face-images"
    output_folder = "/home/vu-lab03-pc24/Downloads/deepfakeimg/clsup"
    process_images_to_deepfake(input_folder, output_folder)