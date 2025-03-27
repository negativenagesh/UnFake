# src/components/mission_section.py

import streamlit as st

def render_mission_section():
    """Render the Our Mission section with the same design as described."""
    mission_html = """
    <div class="mission-section">
        <div class="columns">
            <div class="left-column">
                <div class="image-placeholder top">
                    <div class="icon"></div>
                    <div class="label">Real Image <span class="badge verified">Verified</span></div>
                </div>
                <div class="image-placeholder bottom">
                    <div class="icon"></div>
                    <div class="label">Deepfake <span class="badge detected">Detected</span></div>
                </div>
                <div class="analysis-results">
                    <h3>Analysis Results</h3>
                    <div class="progress-bar">
                        <p>Facial Consistency / Texture Analysis</p>
                        <div class="bar"><div class="fill" style="width: 80%;"></div></div>
                    </div>
                </div>
            </div>
            <div class="right-column">
                <div class="content-block">
                    <div class="icon mission"></div>
                    <h3>Our Mission</h3>
                    <p>The mission is to develop a deepfake image classification system that can accurately identify whether an image is real or a deepfake, thereby protecting users from potential harms such as misinformation, reputation damage, and legal liabilities with high accuracy and reliability</p>
                </div>
                <div class="content-block">
                    <div class="icon challenge"></div>
                    <h3>The Challenge</h3>
                    <p>As deepfake technology becomes more sophisticated, the ability to distinguish between real and manipulated media becomes increasingly difficult, threatening trust in digital content.</p>
                </div>
                <div class="content-block">
                    <div class="icon solution"></div>
                    <h3>Our Solution</h3>
                    <p>We use pretrained CNN models with finetuned on our diverse dataset  of 250000+ images to detect subtle inconsistencies invisible to the human eye</p>
                </div>
            </div>
        </div>
        <div class="decorative-circles">
            <div class="circle top-left"></div>
            <div class="circle bottom-right"></div>
        </div>
    </div>
    """
    return mission_html