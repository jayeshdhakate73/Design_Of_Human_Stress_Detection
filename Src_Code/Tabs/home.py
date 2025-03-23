import streamlit as st
from PIL import Image

def app():
    # Load the image
    image = Image.open("Src_Code/1.png")  # Replace with your actual path

    # Apply General Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Page Background */
        .stApp {
            background-color: #f8f9fa;
            padding: 20px;
        }

        /* Title Styling */
        .title {
            font-size: 2.5rem;
            font-weight: 600;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }

        /* Subtitle Styling */
        .subtitle {
            font-size: 1.3rem;
            font-weight: 400;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }

        /* Section Headers */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 10px;
            text-align: center;
        }

        /* Paragraph Text */
        .paragraph {
            font-size: 1rem;
            line-height: 1.6;
            color: #4a4a4a;
            text-align: center;
            margin-bottom: 20px;
        }

        /* Divider */
        .divider {
            border-top: 2px solid #dcdde1;
            margin: 30px 0;
        }

        /* Button Styling */
        .stButton>button {
            display: block;
            margin: auto;
            font-size: 1.2rem;
            padding: 10px 25px;
            border-radius: 8px;
            background-color: #2ecc71;
            color: white;
            border: none;
        }

        .stButton>button:hover {
            background-color: #27ae60;
            transition: 0.3s ease;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main UI Content
    st.markdown('<div class="title">📊 Human Stress Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🎭 Are You Stressed? Let’s Find Out!</div>', unsafe_allow_html=True)

    # Display Image
    st.image(image, caption="AI-Powered Stress Analysis", use_container_width=True)

    # Welcome Text
    st.markdown(
    '<div class="paragraph">'
    'Welcome to the Stress Detection System, an AI-powered tool that analyzes stress using physiological data and real-time emotion detection. '
    'Leveraging advanced machine learning techniques, it provides accurate insights to help users understand their stress state.'
    '</div>', 
    unsafe_allow_html=True
    )

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Physiological Data Analysis Section
    st.markdown('<div class="section-header">📡 Physiological Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
    '<div class="paragraph">'
    'Our system analyzes physiological data, including GSR, respiration rate, body temperature, and heart rate. '
    'A machine learning model, trained using a Decision Tree classifier, processes this data to detect stress accurately.'
    '</div>', 
    unsafe_allow_html=True
    )

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Live Stress Detection Section
    st.markdown('<div class="section-header">🎭 Live Stress Detection (Emotion Based)</div>', unsafe_allow_html=True)
    st.markdown('<div class="paragraph">Experience real-time stress detection through advanced emotion analysis. Our system monitors and interprets emotional cues to provide immediate feedback on stress levels.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <p style="font-size:14px;">
            Disclaimer: This system provides an estimated stress level based on input data. For medical advice, consult a healthcare professional.
        </p>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    app()
