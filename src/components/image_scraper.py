import streamlit as st
import sys
import os
import time

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.unsplashscrap import scrape_unsplash_images  # Assuming this matches your file structure

def render_image_search():
    """
    Render the image search page with Unsplash image scraping functionality
    """
    st.markdown("""
    <style>
    .image-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        grid-gap: 20px;
        margin: 20px 0;
    }
    .image-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        cursor: pointer;
    }
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(255, 0, 0, 0.2);
    }
    .image-card img {
        width: 100%;
        height: 500px;
        object-fit: cover;
        border-radius: 8px;
    }
    .image-card .download-btn {
        position: absolute;
        bottom: 25px;
        right: 25px;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        border: none;
        padding: 8px 12px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        transition: background-color 0.3s ease;
    }
    .image-card .download-btn:hover {
        background: rgba(255, 0, 0, 0.7);
    }
    .image-author {
        margin-top: 10px;
        font-size: 14px;
        color: #cccccc;
    }
    .load-more-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and search bar
    st.markdown("<h1 style='text-align: center; color: #ff0000;'>Unsplash Image Search</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #cccccc;'>Search and browse images from Unsplash</p>", unsafe_allow_html=True)
    
    # Initialize session state for pagination and image storage
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
        
    if 'all_images' not in st.session_state:
        st.session_state.all_images = []
        
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
        
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""
    
    if 'total_pages' not in st.session_state:
        st.session_state.total_pages = 1
    
    # Search form
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_term = st.text_input("Search for images", value=st.session_state.search_term)
    
    with col2:
        search_button = st.button("Search", use_container_width=True)
        
    if search_button and search_term:
        st.session_state.search_term = search_term
        st.session_state.page_number = 1
        st.session_state.all_images = []
        with st.spinner("Searching images..."):
            try:
                # Get initial results and total pages
                images, total_pages = scrape_unsplash_images(search_term, page=1, per_page=30)
                st.session_state.all_images = images
                st.session_state.total_pages = total_pages
            except Exception as e:
                st.error(f"Error searching images: {str(e)}")
                st.session_state.all_images = []
    
    # Display results if available
    if st.session_state.all_images:
        st.markdown(f"### Results for '{st.session_state.search_term}'")
        
        # Create columns for images (2 per row)
        for i in range(0, len(st.session_state.all_images), 2):
            col1, col2 = st.columns(2)
            
            # First image in row
            if i < len(st.session_state.all_images):
                img = st.session_state.all_images[i]
                with col1:
                    # Create container for image
                    img_container = st.container()
                    
                    # Display image with click handler
                    if img_container.image(img['url'], caption=f"By: {img['author']}", use_container_width=True):
                        st.session_state.selected_image = img
                    
                    # Download button
                    img_container.markdown(f"[Download]({img['download_url']})", unsafe_allow_html=True)
            
            # Second image in row
            if i+1 < len(st.session_state.all_images):
                img = st.session_state.all_images[i+1]
                with col2:
                    # Create container for image
                    img_container = st.container()
                    
                    # Display image with click handler
                    if img_container.image(img['url'], caption=f"By: {img['author']}", use_container_width=True):
                        st.session_state.selected_image = img
                    
                    # Download button
                    img_container.markdown(f"[Download]({img['download_url']})", unsafe_allow_html=True)
        
        # Load More button (centered and smaller)
        with st.container():
            col1, col2, col3 = st.columns([1.5, 1, 1.5])
            with col2:
                if st.button("Load More", key="load_more_btn", use_container_width=True, type="primary"):
                    if st.session_state.page_number < st.session_state.total_pages:
                        st.session_state.page_number += 1
                        with st.spinner("Loading more images..."):
                            try:
                                # Get the next page of results
                                next_page_images, _ = scrape_unsplash_images(
                                    st.session_state.search_term,
                                    page=st.session_state.page_number,
                                    per_page=30
                                )
                                st.session_state.all_images.extend(next_page_images)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading more images: {str(e)}")
                    else:
                        st.info("All images have been loaded.")
    
    elif st.session_state.search_term:
        st.write("No images found. Try a different search term.")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 50px 20px;">
            <h2 style="color: #cccccc;">Start by searching for images</h2>
            <p style="color: #888888;">Enter a keyword in the search box above to find images on Unsplash</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display image details if an image is selected
    if st.session_state.selected_image:
        st.sidebar.title("Image Details")
        st.sidebar.image(st.session_state.selected_image['url'], use_container_width=True)
        st.sidebar.markdown(f"### {st.session_state.selected_image['alt_text']}")
        st.sidebar.markdown(f"**Photographer:** {st.session_state.selected_image['author']}")
        st.sidebar.markdown(f"[Download Full Resolution]({st.session_state.selected_image['download_url']})")
        if st.sidebar.button("Close Details"):
            st.session_state.selected_image = None
            st.rerun()

# Assuming this is called from a main app, otherwise add:
if __name__ == "__main__":
    render_image_search()