import sys
import os

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.app import show_custom_landing_page, setup_page_styling
from src.components.image_scraper import show_image_search_page, show_image_details_page

# Set page configuration
st.set_page_config(
    page_title="UnFake",
    page_icon="UnFake-logo/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def main():
    """Main function to determine which page to display based on query parameters."""
    # Get the current query parameters
    params = st.query_params
    # Extract the 'page' parameter, default to "landing" if not present
    page = params.get("page", "landing")

    # Render the appropriate page based on the query parameter
    if page in ["image_search", "image_scraper"]:
        setup_page_styling()
        show_image_search_page()
    elif page == "details":
        setup_page_styling()
        show_image_details_page()
    else:  # Default to landing page
        show_custom_landing_page()

if __name__ == "__main__":
    main()