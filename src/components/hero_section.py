# src/components/hero_section.py

def render_hero_section():
    """
    Hero section HTML with buttons for demo, custom image testing, and getting started.
    """
    hero_html = """
    <div class="hero-text">
        <h1>Detect DeepFakes</h1>
        <p style="margin-bottom: 0;">Before downloading an image</p>
        <h2>Preserve Truth</h2>
        <p style="margin-bottom: 0;">Our deepfake detection system identifies manipulated facial</p>
        <p style="margin-top: 0; margin-bottom: 0;">images with unprecedented accuracy, helping safeguard digital media</p>
        <p style="margin-top: 0;">authenticity across selected categories of Unsplash images</p>
        <div class="hero-buttons">
            <a href="#powerful_features" class="hero-button explore-btn"
               style="margin-right: 20px; display: inline-block; text-align: center; 
                      padding: 10px 24px; font-weight: bold; border-radius: 5px; 
                      text-decoration: none; background-color: #ff4b4b; color: white;">
               Try Demo
            </a>
            <a href="?page=custom_image" target="_self"
               class="hero-button custom-image-btn"
               style="margin-right: 20px; display: inline-block; text-align: center; 
                      padding: 10px 24px; font-weight: bold; border-radius: 5px; 
                      text-decoration: none; background-color: transparent; color: white;
                      border: 2px solid white;">
               Test Custom Image
            </a>
            <a href="?page=image_search" target="_self"
               class="hero-button how-it-works-btn"
               style="display: inline-block; text-align: center; padding: 10px 24px; 
                      font-weight: bold; border-radius: 5px; text-decoration: none; 
                      background-color: #ff4b4b; color: white;">
               Get Started
            </a>
        </div>
    </div>
    """
    return hero_html