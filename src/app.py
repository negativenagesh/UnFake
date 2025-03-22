import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.components.navbar import render_navbar
from src.components.hero_section import render_hero_section
from src.components.face_detection_preview import render_deepfake_detection
from src.components.image_scraper import show_image_search_page, show_image_details_page

def show_custom_landing_page():
    """Render the landing page."""
    # Page styling
    st.markdown(
        """
        <style>
        body {
            background-color: #262626 !important;
        }
        .stApp {
            background-color: #262626;
            background: radial-gradient(circle, rgba(38, 38, 38, 0.8) 0%, rgba(30, 30, 30, 0.6) 50%, #262626 100%);
        }
        /* Smooth scrolling for anchor links */
        html {
            scroll-behavior: smooth;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    render_navbar()

    # Load custom CSS
    with open("src/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Add an ID for the hero container
    st.markdown('<div id="hero-section"></div>', unsafe_allow_html=True)

    # Two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        hero_html = render_hero_section()
        st.markdown(hero_html, unsafe_allow_html=True)

    with col2:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        face_detection_html = render_deepfake_detection()
        st.markdown(face_detection_html, unsafe_allow_html=True)

    # Scroll arrow
    st.markdown(
        """
        <div style="text-align: center; margin-top: 10px;">
        <a href="#about" id="scroll-arrow" style="text-decoration: none;">
            <span style="font-size: 70px; color: #ffff; transition: transform 0.3s ease; 
                         display: inline-block; margin-top:-50px">↓
            </span>
        </a>
        </div>
        <style>
        #scroll-arrow span:hover { transform: scale(1.2); }
        #scroll-arrow span:active { transform: scale(0.9); }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Keep other sections as needed

def setup_page_styling():
    """Apply shared styling for all pages."""
    st.markdown(
        """
        <style>
        body {
            background-color: #262626 !important;
        }
        .stApp {
            background-color: #262626;
            background: radial-gradient(circle, rgba(38, 38, 38, 0.8) 0%, rgba(30, 30, 30, 0.6) 50%, #262626 100%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    render_navbar()
    
    with open("src/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    """Main function to determine which page to display."""
    params = st.query_params
    page = params.get("page", ["landing"])[0]
    
    # For debugging in terminal:
    print(f"Current page parameter: {page}")
    
    if page == "image_scraper" or page == "image_search":
        print("Loading image search page...")
        setup_page_styling()
        show_image_search_page()
    elif page == "details":
        print("Loading details page...")
        setup_page_styling()
        show_image_details_page()
    else:
        print("Loading landing page...")
        show_custom_landing_page()

if __name__ == "__main__":
    main()