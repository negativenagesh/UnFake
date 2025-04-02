import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.components.navbar import render_navbar
from src.components.hero_section import render_hero_section
from src.components.face_detection_preview import render_deepfake_detection
from src.components.image_scraper import show_image_search_page, show_image_details_page
from src.components.properties import render_about_section
from src.components.mission_section import render_mission_section
from src.components.dataset_section import render_dataset_section
from src.components.use_cases_section import render_use_cases_section
from src.components.footer import render_footer

def apply_global_styling():
    """Apply shared styling for all pages without rendering content."""
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
        html {
            scroll-behavior: smooth;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with open("src/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def show_custom_landing_page():
    """Render the landing page with the new Footer section."""
    render_navbar()

    # Hero section
    st.markdown('<div id="hero-section"></div>', unsafe_allow_html=True)
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

    # About UnFake section
    st.markdown('<div id="about"></div>', unsafe_allow_html=True)
    about_html = render_about_section()
    st.markdown(about_html, unsafe_allow_html=True)

    # Our Mission section
    mission_html = render_mission_section()
    st.markdown(mission_html, unsafe_allow_html=True)

    # Our Dataset section
    dataset_html = render_dataset_section()
    st.markdown(dataset_html, unsafe_allow_html=True)

    # Use Cases section
    use_cases_html = render_use_cases_section()
    st.markdown(use_cases_html, unsafe_allow_html=True)

    # Footer section
    footer_html = render_footer()
    st.markdown(footer_html, unsafe_allow_html=True)

def setup_page_styling():
    """Apply shared styling for all pages (kept for compatibility, but updated)."""
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
    page = params.get("page", "landing")
    
    # Clear analysis state when changing pages
    if page != "details" and 'analysis_performed' in st.session_state:
        st.session_state.analysis_performed = False
        st.session_state.analysis_results = None
    
    if page == "image_scraper" or page == "image_search":
        setup_page_styling()
        show_image_search_page()
    elif page == "details":
        setup_page_styling()
        show_image_details_page()
    else:
        show_custom_landing_page()

if __name__ == "__main__":
    main()