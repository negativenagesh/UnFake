import streamlit as st
from .navbar import render_navbar
from .footer import render_footer

def render_unfake_api_page():
    """Renders the UnFake API usage page."""
    render_navbar()

    # API documentation with visual styling
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background-color: #1a1a1a; margin-bottom: 30px;">
        <h1 style="color: white; text-align: center; margin-top: 25px;">UnFake API Documentation</h1>
        <p style="color: #ddd; text-align: center; font-size: 18px;">
            Integrate deepfake detection directly into your applications
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Overview & Usage", "Code Examples", "Implementation Details"])
    
    with tab1:
        st.markdown("""
        ## Overview
        
        The UnFake API provides a simple and efficient way to integrate deepfake detection into your applications. 
        Our machine learning model has been trained on thousands of images to detect subtle signs of manipulation 
        that are typically invisible to the human eye.
        
        ### Base URL
        ```
        https://api.unfake.ai/v1
        ```
        
        ### Authentication
        API calls require an API key to be included in the request header:
        ```
        X-API-Key: your_api_key_here
        ```
        
        ### Rate Limits
        - Free tier: 100 requests per day
        - Pro tier: 1,000 requests per day
        - Enterprise tier: Custom limits
        
        ## Endpoints
        
        ### Health Check
        ```
        GET /health
        ```
        Returns a status message indicating API availability.
        
        ### Deepfake Detection
        ```
        POST /detect
        ```
        
        **Request Parameters:**
        - File upload or image URL
        
        **Response:**
        ```json
        {
          "is_deepfake": true|false,
          "confidence": 0.97,
          "analysis_time": 0.432,
          "features": [
            "Facial inconsistency detected",
            "Unnatural skin texture",
            "Eye reflection anomalies"
          ]
        }
        ```
        """)
        
        # Add a demo section with sample image analysis
        st.markdown("### Try It Out")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            1. Upload an image
            2. The API will analyze it
            3. View the detection results
            """)
            
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file and st.button("Analyze Image", type="primary"):
                st.info("In a real implementation, this would call the API and display results.")
                # This is for demo purposes only
                st.markdown("""
                ```json
                {
                  "is_deepfake": true,
                  "confidence": 0.97,
                  "analysis_time": 0.432,
                  "features": [
                    "Facial inconsistency detected",
                    "Unnatural skin texture",
                    "Eye reflection anomalies"
                  ]
                }
                ```
                """)
                
        with col2:
            st.markdown("#### How it works")
            st.markdown("""
            1. Image preprocessing
            2. Feature extraction
            3. Deep learning model analysis
            4. Result classification
            
            Our model looks for specific artifacts common in deepfakes:
            - Inconsistent facial features
            - Unnatural skin textures
            - Lighting inconsistencies
            - Blending boundaries
            """)
            
            # Visual representation of the analysis process
            st.markdown("""
            <div style="background-color: #2a2a2a; padding: 10px; border-radius: 5px; margin-top: 20px;">
                <div style="background: linear-gradient(90deg, #ff4b4b 70%, #2a2a2a 70%); height: 30px; border-radius: 3px; margin-bottom: 5px;"></div>
                <div style="color: white; text-align: center;">Confidence: 97% Deepfake</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        ## Code Examples
        
        ### Python
        ```python
        import requests
        
        def detect_deepfake(image_path, api_key):
            url = "https://api.unfake.ai/v1/detect"
            headers = {
                "X-API-Key": api_key
            }
            
            with open(image_path, "rb") as image_file:
                files = {"file": image_file}
                response = requests.post(url, headers=headers, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Deepfake: {result['is_deepfake']}")
                print(f"Confidence: {result['confidence']}")
                print(f"Analysis time: {result['analysis_time']}s")
                print("Features detected:")
                for feature in result['features']:
                    print(f"- {feature}")
                return result
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
        
        # Example usage
        api_key = "your_api_key_here"
        result = detect_deepfake("path/to/image.jpg", api_key)
        ```
        
        ### JavaScript
        ```javascript
        async function detectDeepfake(imageFile, apiKey) {
            const url = "https://api.unfake.ai/v1/detect";
            const formData = new FormData();
            formData.append("file", imageFile);
            
            try {
                const response = await fetch(url, {
                    method: "POST",
                    headers: {
                        "X-API-Key": apiKey
                    },
                    body: formData
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log(`Deepfake: ${result.is_deepfake}`);
                    console.log(`Confidence: ${result.confidence}`);
                    console.log(`Analysis time: ${result.analysis_time}s`);
                    console.log("Features detected:");
                    result.features.forEach(feature => {
                        console.log(`- ${feature}`);
                    });
                    return result;
                } else {
                    console.error(`Error: ${response.status}`);
                    console.error(await response.text());
                    return null;
                }
            } catch (error) {
                console.error("Request failed:", error);
                return null;
            }
        }
        
        // Example usage
        const apiKey = "your_api_key_here";
        const fileInput = document.getElementById("fileInput");
        fileInput.addEventListener("change", async (event) => {
            const file = event.target.files[0];
            if (file) {
                const result = await detectDeepfake(file, apiKey);
                // Process result...
            }
        });
        ```
        
        ### cURL
        ```bash
        curl -X POST \\
          -H "X-API-Key: your_api_key_here" \\
          -F "file=@/path/to/image.jpg" \\
          https://api.unfake.ai/v1/detect
        ```
        """)
    
    with tab3:
        st.markdown("""
        ## Setting Up Your Own UnFake API Server
        
        You can deploy the UnFake API on your own server for enhanced privacy or custom integration.
        
        ### Requirements
        - Python 3.8+
        - FastAPI
        - PyTorch 1.10+
        - CUDA (optional, for GPU acceleration)
        
        ### Installation
        
        1. Clone the repository:
        ```bash
        git clone https://github.com/negativenagesh/UnFake-API.git
        cd UnFake-API
        ```
        
        2. Install dependencies:
        ```bash
        pip install -r requirements.txt
        ```
        
        3. Start the API server:
        ```bash
        MODEL_PATH=/path/to/model.pth uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        ```
        
        4. Visit the interactive API documentation:
        ```
        http://localhost:8000/docs
        ```
        
        ### Core Files
        
        #### main.py
        This is the entry point of the API that sets up FastAPI and defines endpoints.
        
        #### model.py
        Defines the PyTorch model architecture for deepfake detection.
        
        #### schemas.py
        Contains Pydantic models for request and response validation.
        
        #### transforms.py
        Handles image preprocessing and transformations.
        
        ### Docker Deployment
        
        ```bash
        # Build the Docker image
        docker build -t unfake-api .
        
        # Run the container
        docker run -p 8000:8000 -e MODEL_PATH=/app/models/deepfake_model.pth unfake-api
        ```
        """)

    # Get API key section
    st.markdown("""
    ## Get API Access
    
    To get started with the UnFake API, request an API key by filling out the form below:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Name")
        st.text_input("Email")
        st.selectbox("Usage Type", ["Personal", "Commercial", "Educational", "Research"])
    with col2:
        st.text_area("Intended Use Case")
        st.number_input("Estimated API calls per month", min_value=1, value=100)
        
    if st.button("Request API Access", type="primary"):
        st.success("Thank you! We'll review your request and contact you with your API key.")

    # Inject custom CSS for API page
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    # Footer
    st.markdown(render_footer(), unsafe_allow_html=True)
