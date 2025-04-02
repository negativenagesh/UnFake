# components/navbar.py
import streamlit as st
import os

def render_navbar():
    # Load CSS
    with open("src/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    logo_path = "UnFake-logo/logo.png"
    
    # Navbar HTML with anchor links for navigation
    navbar_html = f"""
    <div class="navbar">
        <div class="navbar-logo">
            <a href="/">
                <img src="data:image/png;base64,{get_image_as_base64(logo_path)}" height="150">
            </a>
        </div>
        <div class="navbar-links">
            <a href="#hero-section" style="font-size: 18px;">Home</a>
            <a href="#why_resumai_exists" style="font-size: 18px;">About</a>
            <a href="#powerful_features" style="font-size: 18px;">Technology</a>
            <a href="#how_resumai_works" style="font-size: 18px;">Features</a>
            <a href="#resumai_benefits" style="font-size: 18px;">Dataset</a>
            <a href="#success_stories" style="font-size: 18px;">Demo</a>
            <a href="#faq" style="font-size: 18px;">Use Cases</a>
            <a href="" style="font-size: 18px;">Research</a>
            <a href="https://github.com/negativenagesh/UnFake" target="_blank">
                <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" style="width: 30px; height: 30px; margin-left: 10px;">
            </a>
        </div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)

def get_image_as_base64(path):
    """Convert an image file to base64 string"""
    import base64
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')