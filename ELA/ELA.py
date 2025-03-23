from PIL import Image

def error_level_analysis(image_path, quality=95):
    """
    Performs Error Level Analysis (ELA) on the given image.
    
    Args:
        image_path (str): Path to the input image.
        quality (int): JPEG quality level for resaving (default is 95).
    
    Returns:
        PIL.Image: Grayscale ELA image where brighter areas indicate higher error levels.
    """
    # Load the original image and convert to RGB mode
    original = Image.open(image_path).convert('RGB')
    
    # Save the image as JPEG with the specified quality to introduce compression artifacts
    resaved_path = 'resaved.jpg'
    original.save(resaved_path, 'JPEG', quality=quality)
    
    # Load the resaved JPEG image
    resaved = Image.open(resaved_path).convert('RGB')
    
    # Create a new grayscale image for the ELA result
    ela = Image.new('L', original.size)
    
    # Access pixels from original, resaved, and ELA images
    pixels_orig = original.load()
    pixels_resaved = resaved.load()
    pixels_ela = ela.load()
    
    max_diff = 0
    
    # Compute the maximum difference across RGB channels for each pixel
    for x in range(original.width):
        for y in range(original.height):
            r1, g1, b1 = pixels_orig[x, y]
            r2, g2, b2 = pixels_resaved[x, y]
            dr = abs(r1 - r2)
            dg = abs(g1 - g2)
            db = abs(b1 - b2)
            ela_value = max(dr, dg, db)  # Take the maximum difference
            pixels_ela[x, y] = ela_value
            if ela_value > max_diff:
                max_diff = ela_value
    
    # Scale the ELA image to enhance visibility
    if max_diff == 0:
        print("No difference detected between original and resaved images.")
        return ela  # Return unscaled if images are identical
    
    scale = 255.0 / max_diff
    for x in range(original.width):
        for y in range(original.height):
            pixels_ela[x, y] = int(pixels_ela[x, y] * scale)
    
    return ela

# Example usage
# Replace 'input.jpg' with the path to your image
ela_image = error_level_analysis('/home/subrahmanya/projects/UnFake/deep-fake-images-generated/oldman-to-women.jpg')
ela_image.save('ela_output7.png')