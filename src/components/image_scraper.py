import streamlit as st
import sys
import os
import time

# Add backend to path (adjust based on your structure)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.components.navbar import render_navbar  # Added import
from backend.unsplashscrap import scrape_unsplash_images  # Ensure this matches your file structure

def show_landing_page():
    """Display the landing page."""
    render_navbar()  # Added
    st.title("Welcome to UnFake")
    st.write("This is the landing page.")
    if st.button("Go to Image Search"):
        st.session_state.page = "image_scraper"
        st.query_params["page"] = "image_search"
        st.rerun()

def show_image_search_page():
    """Render the image search page with Unsplash image scraping functionality."""
    render_navbar()  # Added
    # Custom CSS for the search page
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
    .load-more-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    if 'all_images' not in st.session_state:
        st.session_state.all_images = []
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""
    if 'total_pages' not in st.session_state:
        st.session_state.total_pages = 1
    if 'last_search' not in st.session_state:
        st.session_state.last_search = ""
    
    st.markdown("<br> </br>", unsafe_allow_html=True)

    # Search form
    with st.form(key="search_form"):
        col1, col2 = st.columns([4, 0.5])
        with col1:
            search_term = st.text_input("Search for images", value=st.session_state.search_term)
        with col2:
            search_button = st.form_submit_button("Search", use_container_width=True)

        
        if search_button and search_term:
            st.session_state.search_term = search_term
            st.session_state.page_number = 1
            st.session_state.all_images = []
            st.session_state.selected_image = None
    
    # Perform search
    if search_term and search_term != st.session_state.last_search:
        with st.spinner("Searching images..."):
            try:
                images, total_pages = scrape_unsplash_images(search_term, page=1, per_page=30)
                st.session_state.all_images = images
                st.session_state.total_pages = total_pages
                st.session_state.last_search = search_term
            except Exception as e:
                st.error(f"Error searching images: {str(e)}")
                st.session_state.all_images = []
    
    # Display results
    if st.session_state.all_images:
        st.markdown(f"### Results for '{st.session_state.search_term}'")
        
        for i in range(0, len(st.session_state.all_images), 2):
            col1, col2 = st.columns(2)
            
            # First image
            if i < len(st.session_state.all_images):
                img = st.session_state.all_images[i]
                with col1:
                    with st.container():
                        st.image(img['display_url'], caption=f"By: {img['author_name']} (@{img['author_username']})", use_container_width=True)
                        if st.button("View Details", key=f"view_{img['id']}_1"):
                            st.session_state.selected_image = img
                            # Reset fake detection result when selecting a new image
                            if 'fake_detection_result' in st.session_state:
                                st.session_state.fake_detection_result = None
                            st.session_state.page = "details"
                            st.query_params["page"] = "details"
                            st.rerun()
                        st.markdown(f"[Download]({img['download_url']})", unsafe_allow_html=True)
            
            # Second image
            if i + 1 < len(st.session_state.all_images):
                img = st.session_state.all_images[i + 1]
                with col2:
                    with st.container():
                        st.image(img['display_url'], caption=f"By: {img['author_name']} (@{img['author_username']})", use_container_width=True)
                        if st.button("View Details", key=f"view_{img['id']}_2"):
                            st.session_state.selected_image = img
                            # Reset fake detection result when selecting a new image
                            if 'fake_detection_result' in st.session_state:
                                st.session_state.fake_detection_result = None
                            st.session_state.page = "details"
                            st.query_params["page"] = "details"
                            st.rerun()
                        st.markdown(f"[Download]({img['download_url']})", unsafe_allow_html=True)
        
        # Load More button
        with st.container():
            col1, col2, col3 = st.columns([1.5, 1, 1.5])
            with col2:
                if st.button("Load More", key="load_more_btn", use_container_width=True, type="primary"):
                    if st.session_state.page_number < st.session_state.total_pages:
                        st.session_state.page_number += 1
                        with st.spinner("Loading more images..."):
                            try:
                                next_page_images, _ = scrape_unsplash_images(
                                    st.session_state.search_term,
                                    page=st.session_state.page_number,
                                    per_page=30
                                )
                                st.session_state.all_images.extend(next_page_images)
                                st.session_state.selected_image = None
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

def show_image_details_page():
    """Display the details of the selected image."""
    render_navbar()  # Added
    # Custom CSS for the details page
    st.markdown("""
    <style>
    .selected-image {
        margin-top: 20px;
        padding: 20px;
        background-color: #1a1a1a;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .selected-image img {
        width: 100%;
        max-width: 800px;
        margin: 0 auto;
        display: block;
        border-radius: 8px;
    }
    .selected-image-details {
        margin-top: 15px;
        font-size: 16px;
        color: #cccccc;
    }
    .selected-image-details strong {
        color: #ff0000;
    }
    .fake-check-result {
        margin-top: 20px;
        padding: 15px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .fake-result {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .real-result {
        background-color: #ccffcc;
        color: #006600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if a selected image exists
    if 'selected_image' not in st.session_state or st.session_state.selected_image is None:
        st.error("No image selected. Please go back and select an image.")
        if st.button("Back to Search", key="back_btn_top"):
            # Reset fake detection result when going back
            if 'fake_detection_result' in st.session_state:
                st.session_state.fake_detection_result = None
            st.session_state.page = "image_scraper"
            st.query_params["page"] = "image_search"
            st.rerun()
        return
    
    # Initialize fake detection result in session state if not exists
    if 'fake_detection_result' not in st.session_state:
        st.session_state.fake_detection_result = None
    
    # Display the selected image and its details
    selected = st.session_state.selected_image
    
    st.subheader("Image Details")
    
    # Single column layout for the image and details
    st.markdown("<div class='selected-image'>", unsafe_allow_html=True)
    
    # Image
    st.image(selected['display_url'], use_container_width=True)
    
    # Back button above the DeepFake check button
    if st.button("Back to Search", key="back_btn_top"):
        # Reset fake detection result when going back
        if 'fake_detection_result' in st.session_state:
            st.session_state.fake_detection_result = None
        st.session_state.page = "image_scraper"
        st.query_params["page"] = "image_search"
        st.rerun()
    
    # DeepFake check button below the image
    if st.button("Check if DeepFake", use_container_width=True, type="primary", key="deepfake_btn"):
        with st.spinner("Analyzing image..."):
            try:
                # Import the backend function for fake detection
                from backend.deepfake_detection import analyze_image_for_streamlit
                
                # Call the safe wrapper function
                image_url = selected['display_url']
                result = analyze_image_for_streamlit(image_url)
                
                # Store the result in session state
                st.session_state.fake_detection_result = result
                st.rerun()  # Rerun to display the result
                
            except Exception as e:
                st.error(f"Error during fake detection: {str(e)}")
                import traceback
                st.exception(traceback.format_exc())
    
    # Display fake detection result if available
    if st.session_state.fake_detection_result is not None:
        result = st.session_state.fake_detection_result
        is_fake = result.get("is_fake", False)
        confidence = result.get("confidence", 0.0)
        message = result.get("message", "")
        
        st.markdown("### Analysis Result")
        if is_fake:
            st.error(f"⚠️ This image appears to be **FAKE**")
        else:
            st.success(f"✅ This image appears to be **REAL**")
        
        st.metric("Confidence", f"{confidence*100:.1f}%")
        
        if message:
            st.info(message)
    
    # Image details
    st.markdown("<div class='selected-image-details'>", unsafe_allow_html=True)
    st.write(f"**Alt Text:** {selected.get('alt_text', 'N/A')}")
    st.write(f"**Photographer:** {selected.get('author_name', 'N/A')} (@{selected.get('author_username', 'N/A')})")
    st.write(f"**Dimensions:** {selected.get('width', 'N/A')} x {selected.get('height', 'N/A')}")
    st.write(f"**Created At:** {selected.get('created_at', 'N/A')}")
    st.write(f"**Likes:** {selected.get('likes', 'N/A')}")
    st.write(f"**Dominant Color:** {selected.get('color', 'N/A')}")
    st.markdown(f"[Download Full Resolution]({selected.get('download_url', '#')})")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Back button at the bottom
    if st.button("Back to Search", key="back_btn_bottom"):
        # Reset fake detection result when going back
        if 'fake_detection_result' in st.session_state:
            st.session_state.fake_detection_result = None
        st.session_state.page = "image_scraper"
        st.query_params["page"] = "image_search"
        st.rerun()

def show_custom_image_page():
    """Display the custom image upload page and analyze uploaded images."""
    render_navbar()  # Added
    st.title("Test Custom Image")
    st.markdown("Upload any image to check if it's a deepfake")
    
    # File uploader for custom images
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to run the analysis
        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Save the uploaded file temporarily
                    temp_path = f"temp_upload_{int(time.time())}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Import the backend function for fake detection
                    from backend.deepfake_detection import analyze_image_for_streamlit
                    
                    # Call the analysis function
                    result = analyze_image_for_streamlit(temp_path)
                    
                    # Display results
                    st.markdown("### Analysis Result")
                    if result.get("is_fake", False):
                        st.error(f"⚠️ This image appears to be FAKE with {result.get('confidence', 0)*100:.1f}% confidence")
                        
                        # Show additional analysis results if available
                        if "message" in result and result["message"]:
                            st.info(result["message"])
        
                    else:
                        st.success(f"✅ This image appears to be REAL with {result.get('confidence', 0)*100:.1f}% confidence")
                    
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
                except Exception as e:
                    st.error(f"Error during fake detection: {str(e)}")
                    import traceback
                    st.exception(traceback.format_exc())
    
    # Back button
    if st.button("Back to Home"):
        st.session_state.page = "landing"
        st.query_params.clear()
        st.rerun()