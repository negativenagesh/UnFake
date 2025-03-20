import streamlit as st
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.components.navbar import render_navbar
from src.components.hero_section import render_hero_section
from src.components.face_detection_preview import render_deepfake_detection 

def show_landing_page():
    """Render the landing page."""
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
    
    with open("src/styles/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    

    # Add an ID for the hero section container
    st.markdown('<div id="hero-section"></div>', unsafe_allow_html=True)

    # Create a container for the hero section and resume preview
    col1, col2 = st.columns([1, 1], gap="large")  # Equal width columns with spacing

    with col1:
    # Add some vertical margin/padding for better alignment
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        hero_html = render_hero_section()
        st.markdown(hero_html, unsafe_allow_html=True)

    with col2:
    # Add some vertical margin/padding for better alignment
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        face_detection_html = render_deepfake_detection()
        st.markdown(face_detection_html, unsafe_allow_html=True)

    # Add a downward arrow at the bottom (simulating "scroll down")
    st.markdown(
        """
        <div style="text-align: center; margin-top: 10px;">
        <a href="#about" id="scroll-arrow" style="text-decoration: none;">
            <span style="font-size: 70px; color: #ffff; transition: transform 0.3s ease; display: inline-block;">↓</span>
        </a>
        </div>
        <style>
        #scroll-arrow span:hover {
            transform: scale(1.2);
        }
        #scroll-arrow span:active {
            transform: scale(0.9);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # # Add IDs for each section for anchor navigation
    # st.markdown('<div id="why_resumai_exists"></div>', unsafe_allow_html=True)
    # why_resumai_html = render_why_resumai_exists()
    # st.markdown(why_resumai_html, unsafe_allow_html=True)

    # st.markdown('<div id="powerful_features"></div>', unsafe_allow_html=True)
    # powerful_features_html = render_powerful_features()
    # st.markdown(powerful_features_html, unsafe_allow_html=True)

    # st.markdown('<div id="how_resumai_works"></div>', unsafe_allow_html=True)
    # how_resumai_works_html = render_how_resumai_works()
    # st.markdown(how_resumai_works_html, unsafe_allow_html=True)

    # st.markdown('<div id="ready_to_optimize"></div>', unsafe_allow_html=True)
    # ready_to_optimize_html = render_ready_to_optimize()
    # st.markdown(ready_to_optimize_html, unsafe_allow_html=True)

    # st.markdown('<div id="why_choose_resumai"></div>', unsafe_allow_html=True)
    # why_choose_resumai_html = render_why_choose_resumai()
    # st.markdown(why_choose_resumai_html, unsafe_allow_html=True)

    # st.markdown('<div id="resumai_benefits"></div>', unsafe_allow_html=True)
    # resumai_benefits_html = render_resumai_benefits()
    # st.markdown(resumai_benefits_html, unsafe_allow_html=True)

    # st.markdown('<div id="success_stories"></div>', unsafe_allow_html=True)
    # success_stories_html = render_success_stories()
    # st.markdown(success_stories_html, unsafe_allow_html=True)

    # st.markdown('<div id="resumai_difference"></div>', unsafe_allow_html=True)
    # resumai_difference_html = render_resumai_difference()
    # st.markdown(resumai_difference_html, unsafe_allow_html=True)

    # st.markdown('<div id="faq"></div>', unsafe_allow_html=True)
    # faq_html = render_faq()
    # st.markdown(faq_html, unsafe_allow_html=True)

    # st.markdown('<div id="get_in_touch"></div>', unsafe_allow_html=True)
    # get_in_touch_html = render_get_in_touch()
    # st.markdown(get_in_touch_html, unsafe_allow_html=True)

    # st.markdown('<div id="footer"></div>', unsafe_allow_html=True)
    # footer_html = render_footer()
    # st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    show_landing_page()