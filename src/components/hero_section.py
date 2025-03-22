# ...existing code...
def render_hero_section():
    """
    Hero section HTML with 'Get Started' button linking to '?page=image_search'.
    Removing the 'onclick' and keeping only 'href' + 'target="_self"' will help
    ensure correct navigation in the same tab.
    """
    hero_html = """
    <div class="hero-text">
        <h1>Detect DeepFakes</h1>
        <h2>Preserve Truth</h2>
        <p style="margin-bottom: 0;">Our AI-powered deepfake detection system identifies manipulated facial</p>
        <p style="margin-top: 0; margin-bottom: 0;">images with unprecedented accuracy, helping safeguard digital media</p>
        <p style="margin-top: 0;">authenticity across selected categories of Unsplash images</p>
        <div class="hero-buttons">
            <a href="#powerful_features" class="hero-button explore-btn"
               style="margin-right: 20px;">
               Try Demo
            </a>
            <a href="?page=image_search" target="_self"
               class="hero-button how-it-works-btn"
               style="display: inline-block; text-align: center; padding: 10px 24px; 
                      font-weight: bold; border-radius: 5px; text-decoration: none; 
                      background-color: #ff4b4b; color: white;">
               Get Started
            </a>
        </div>
        <div class="hero-tags">
            <p>Powered by Unsplash API, Zyte, DNNS, computer vision</p>
        </div>
    </div>
    """
    return hero_html