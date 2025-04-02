# deep_fake_detection.py
import streamlit as st

def render_deepfake_detection():
    # Deepfake detection HTML
    deepfake_html = """
    <div class="deepfake-container">
        <div class="face-outline">
            <div class="face-eyes">
                <div class="eye left"></div>
                <div class="eye right"></div>
            </div>
            <div class="face-mouth"></div>
            <div class="face-border"></div>
        </div>
        <div class="deepfake-result">
            <span class="result-text">DEEPFAKE DETECTED</span>
            <span class="confidence-text">CONFIDENCE: 96.7%</span>
        </div>
        <div class="version-text">UnFake v0.1.0</div>
    </div>
    """

    # Inject custom CSS
    with open("src/styles/styles.css", "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

    # Render the HTML
    st.markdown(deepfake_html, unsafe_allow_html=True)

# Call the function to render the component
if __name__ == "__main__":
    render_deepfake_detection()