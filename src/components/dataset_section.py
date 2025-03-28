# src/components/dataset_section.py

import streamlit as st

def render_dataset_section():
    """Render the Our Dataset section with the same design as described."""
    dataset_html = """
    <div class="dataset-section">
        <div class="header">
            <h1 class="title">Our <span class="highlight">Dataset</span></h1>
            <div class="underline"></div>
            <p class="subtitle">Comprehensive collection of 250000+ images for robust deepfake detection</p>
        </div>
        <div class="columns">
            <div class="left-column">
                <h3>Dataset Statistics</h3>
                <div class="stat-block">
                    <div class="icon total-images"></div>
                    <div class="stat-content">
                        <p class="stat-label">Total Images</p>
                        <p class="stat-value">250000+</p>
                    </div>
                </div>
                <div class="stat-block">
                    <div class="icon ethnic-groups"></div>
                    <div class="stat-content">
                        <p class="stat-label">Ethnic Categories</p>
                        <p class="stat-value">3 Major Groups</p>
                    </div>
                </div>
                <div class="stat-block">
                    <div class="icon feature-types"></div>
                    <div class="stat-content">
                        <p class="stat-label">Feature Categories</p>
                        <p class="stat-value">5+ Major Types</p>
                    </div>
                </div>
            </div>
            <div class="right-column">
                <h3>Category Distribution</h3>
                <div class="distribution-block">
                    <p>General Human Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #FF5252;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Asian Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #FF9800;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Black Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #FFEB3B;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Caucasian Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #CDDC39;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Beard Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #4CAF50;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Freckless Face</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #009688;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Wrinkled Face</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #00BCD4;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Spectacles Face</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #2196F3;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Closeup/Headshots/Portraits Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #3F51B5;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Child Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #9C27B0;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Feature-specific Faces</p>
                    <div class="bar"><div class="fill" style="width: 8.33%; background-color: #E91E63;"></div></div>
                    <span class="percentage">8.33%</span>
                </div>
                <div class="distribution-block">
                    <p>Composite-specific</p>
                    <div class="bar"><div class="fill" style="width: 8.37%; background-color: #795548;"></div></div>
                    <span class="percentage">8.37%</span>
                </div>
            </div>
        </div>
    </div>
    """
    return dataset_html