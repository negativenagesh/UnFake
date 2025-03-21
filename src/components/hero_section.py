import streamlit as st
import sys
import os

# Adjust the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def render_hero_section():
    # Hero section HTML with Get Started button triggering image search page navigation
    hero_html = """
    <div class="hero-text">
        <h1>Detect DeepFakes</h1>
        <h2>Preserve Truth</h2>
        <p style="margin-bottom: 0;">Our AI-powered deepfake detection system identifies manipulated facial</p>
        <p style="margin-top: 0; margin-bottom: 0;">images with unprecedented accuracy, helping safeguard digital media</p>
        <p style="margin-top: 0;">authenticity across selected categories of Unsplash images</p>
        <div class="hero-buttons">
            <a href="#powerful_features" class="hero-button explore-btn" onclick="parent.location='#powerful_features'">Try Demo</a>
            <a href="/?page=image_search" class="hero-button how-it-works-btn">Get Started</a>
        </div>
        <div class="hero-tags">
            <p>Powered by Unsplash API, Zyte, DNNS, computer vision</p>
        </div>
    </div>
    """
    
    return hero_html