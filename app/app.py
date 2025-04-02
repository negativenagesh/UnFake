import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.app import show_custom_landing_page, apply_global_styling
from src.components.image_scraper import show_image_search_page, show_image_details_page, show_custom_image_page
from src.components.unfake_api import render_unfake_api_page

# Set page configuration
st.set_page_config(
    page_title="UnFake",
    page_icon="UnFake-logo/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main function to determine which page to display based on query parameters."""
    # Apply global styling once
    apply_global_styling()
    
    # Get the page from query parameters
    query_params = st.query_params
    page = query_params.get("page", ["landing"])[0] if isinstance(query_params.get("page", ["landing"]), list) else query_params.get("page", "landing")
    
    # Set the page in session state
    if "page" not in st.session_state:
        st.session_state.page = page
    elif page != "landing":  # Only update if coming from a direct link
        st.session_state.page = page
    
    # Route to appropriate page - each function renders its own navbar and content
    if st.session_state.page == "image_search":
        show_image_search_page()
    elif st.session_state.page == "details":
        show_image_details_page()
    elif st.session_state.page == "custom_image":
        show_custom_image_page()
    elif page == "unfake_api":
        render_unfake_api_page()
    else:
        show_custom_landing_page()

if __name__ == "__main__":
    main()