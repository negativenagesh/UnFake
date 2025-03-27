# src/components/about_section.py

import streamlit as st

def render_about_section():
    """Render the About UnFake section."""
    about_html = """
    <div class="about-section">
        <div class="stats-container">
            <div class="stat-box">
                <h2 class="stat-number">250000+</h2>
                <p class="stat-description">Images in Dataset</p>
            </div>
            <div class="stat-box">
                <h2 class="stat-number">97.3%</h2>
                <p class="stat-description">Detection Accuracy</p>
            </div>
            <div class="stat-box">
                <h2 class="stat-number">5+</h2>
                <p class="stat-description">Ethnic Categories</p>
            </div>
            <div class="stat-box">
                <h2 class="stat-number">500ms</h2>
                <p class="stat-description">Processing Time</p>
            </div>
        </div>
        <h1 class="about-title">About <span class="highlight">UnFake</span></h1>
        <p class="about-tagline">Distinguishing reality from fabrication in the digital age</p>
    </div>
    """
    return about_html