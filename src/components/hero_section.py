def render_hero_section():
    """
    Hero section HTML with 'Get Started' button linking to '?page=image_search'.
    Removing the 'onclick' and keeping only 'href' + 'target="_self"' will help
    ensure correct navigation in the same tab.
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
            <a href="?page=image_search" target="_self"
               class="hero-button how-it-works-btn"
               style="margin-right: 20px; display: inline-block; text-align: center; padding: 10px 24px; 
                      font-weight: bold; border-radius: 5px; text-decoration: none; 
                      background-color: #ff4b4b; color: white;">
               Get Started
            </a>
            <a href="?page=custom_image" target="_self"
               class="hero-button custom-image-btn"
               style="margin-right: 20px; display: inline-block; text-align: center; padding: 10px 24px; 
                      font-weight: bold; border-radius: 5px; text-decoration: none; 
                      background-color: #ff4b4b; color: white;">
               Test Custom Image
            </a>
            <a href="?page=unfake_api" target="_self"
               class="hero-button custom-image-btn"
               style="display: inline-block; text-align: center; padding: 10px 24px; 
                      font-weight: bold; border-radius: 5px; text-decoration: none; 
                      background-color: #ff4b4b; color: white;">
               UnFake API
            </a>            
        </div>
    </div>
    """
    return hero_html