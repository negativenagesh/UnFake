import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.app import show_landing_page, show_image_search_page

st.set_page_config(
    page_title="UnFake",
    page_icon="UnFake-logo/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    params = st.query_params
    if "page" in params:
        page_param = params["page"]
        page_mapping = {
            "app": "main_app",
            "landing": "landing",
            "auth": "auth",
            "image_scraper": "image_scraper",
            "image_search": "image_scraper"  # Added for compatibility
        }
        if page_param in page_mapping:
            st.session_state.page = page_mapping[page_param]

    if st.session_state.page == "landing":
        show_landing_page()
    elif st.session_state.page == "image_scraper":
        show_image_search_page()

if __name__ == "__main__":
    main()