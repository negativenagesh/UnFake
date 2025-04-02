import streamlit as st
from .navbar import render_navbar  # Adjust imports if needed
from .footer import render_footer  # Adjust imports if needed

def render_unfake_api_page():
    """Renders the UnFake API usage page."""
    render_navbar()  # Show your existing navbar at the top

    st.title("UnFake API Documentation")

    st.markdown("""
    This API allows you to detect deepfake images using your trained PyTorch model.

    **Base URL**: `http://your-server-ip:8000`

    **Endpoints**:

    1. **Health Check**  
       `GET /`  
       Returns a simple message indicating the API is running.

    2. **Deepfake Detection**  
       `POST /detect`  
       Accepts an image file and returns JSON with:
       - **is_deepfake**: Boolean (true/false)
       - **confidence**: Confidence score (0 to 1)
       - **analysis_time**: In seconds
       - **features**: List of deepfake artifacts found

    ### Example cURL Usage
    ```bash
    curl -X POST -F "file=@/path/to/image.jpg" http://your-server-ip:8000/detect
    ```

    ### How to Run:
    1. Make sure you have installed the required libraries (`fastapi`, `uvicorn`, `torch`, `PIL`, etc.).
    2. Start the server:
       ```bash
       uvicorn main:app --host 0.0.0.0 --port 8000 --reload
       ```
    3. Visit `http://localhost:8000/docs` for interactive docs (provided by FastAPI).
    
    ---
    """)

    # Footer
    render_footer()