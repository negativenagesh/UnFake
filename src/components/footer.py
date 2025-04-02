# src/components/footer.py

import streamlit as st
import os
import base64

def get_image_as_base64(path):
    """Convert an image file to base64 string."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def render_footer():
    """Render the footer section with the same design as previous sections."""
    # Path to the logo (you'll need to place your logo in this directory)
    logo_path = "UnFake-logo/logo.png"

    footer_html = f"""
    <div class="footer-container">
        <div class="footer-content">
            <div class="footer-column">
                <a href="/">
                    <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" height="50" alt="UnFake" style="margin-bottom: 20px;">
                </a>
                <p>Distinguishing reality from fabrication in the digital age with advanced deepfake detection technology.</p>
                <div class="social-icons">
                    <a href="https://github.com/negativenagesh/UnFake" target="_blank">
                        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width: 25px; height: 25px; margin-left: 10px;">
                    </a>
                    <a href="https://x.com/_subrahmanya_" target="_blank">
                        <img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" alt="Twitter" style="width: 25px; height: 25px; margin-left: 10px;">
                    </a>
                    <a href="https://www.linkedin.com/in/subrahmanya-gaonkar/" target="_blank">
                        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn" style="width: 25px; height: 25px; margin-left: 10px;">
                    </a>
                </div>
            </div>
            <div class="footer-column">
                <h3>Quick Links</h3>
                <ul>
                    <li><a href="#hero-section">Home</a></li>
                    <li><a href="#about">About</a></li>
                    <li><a href="#mission">Mission</a></li>
                    <li><a href="#dataset">Dataset</a></li>
                    <li><a href="#use-cases">Use Cases</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </div>
            <div class="footer-column">
                <h3>Resources</h3>
                <ul>
                    <li><a href="https://github.com/negativenagesh/UnFake" target="_blank">GitHub Repository</a></li>
                    <li><a href="https://github.com/negativenagesh/UnFake/blob/main/README.md" target="_blank">Documentation</a></li>
                    <li><a href="https://github.com/negativenagesh/UnFake/blob/main/README.md#%EF%B8%8F-setup" target="_blank">Setup Guide</a></li>
                    <li><a href="https://github.com/negativenagesh/UnFake" target="_blank">Demo</a></li>
                    <li><a href="https://github.com/negativenagesh/UnFake/pulls" target="_blank">Contributing</a></li>
                    <li><a href="https://github.com/negativenagesh/UnFake/blob/main/LICENSE" target="_blank">License</a></li>
                </ul>
            </div>
            <div class="footer-column">
                <h3>Stay Updated</h3>
                <p>Subscribe to our newsletter for the latest updates and insights on deepfake detection.</p>
                <div class="newsletter-form" style="display: flex; flex-direction: column; width: 100%;">
                    <input type="email" placeholder="Your email" style="margin-bottom: 5px; padding: 8px; width: 100%; box-sizing: border-box;">
                    <button style="width: 100%; padding: 8px; box-sizing: border-box;">Subscribe</button>
                </div>
                <p class="newsletter-note">By subscribing, you agree to our Privacy Policy and consent to receive updates.</p>
            </div>
        </div>
        <div class="footer-bottom">
            <span>© 2025 UnFake. All rights reserved.</span>
            <div>
                <a href="#">Privacy Policy</a>
                <a href="#">Terms of Service</a>
                <a href="#">Cookie Policy</a>
            </div>
        </div>
    </div>
    """
    return footer_html