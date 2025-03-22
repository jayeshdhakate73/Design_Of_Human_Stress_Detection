import streamlit as st

def app():
    # Elegant CSS Styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }

        /* Main container */
        .main-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Title styling */
        .title {
            font-size: 2.5rem;
            font-weight: 600;
            color: #2d3436;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        /* Subtitle styling */
        .subtitle {
            font-size: 1.4rem;
            font-weight: 400;
            color: #636e72;
            text-align: center;
            margin-bottom: 2.5rem;
        }

        /* Section header styling */
        .section-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2d3436;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        /* Paragraph styling */
        .paragraph {
            font-size: 1rem;
            line-height: 1.6;
            color: #4a4a4a;
            margin-bottom: 1.5rem;
        }

        /* Divider styling */
        .divider {
            border-top: 1px solid #e0e0e0;
            margin: 2rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Main Content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Title and Subtitle
    st.markdown('<div class="title">Stress Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Advanced Stress Analysis & Monitoring</div>', unsafe_allow_html=True)

    # Welcome Section
    st.markdown(
        """
        <div class="paragraph">
            Welcome to the Stress Detection System, a comprehensive platform designed to analyze and monitor stress levels using both physiological data and real-time emotion detection. This system provides valuable insights into stress patterns, enabling users to take proactive steps towards better mental well-being.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Physiological Data Analysis Section
    st.markdown('<div class="section-header">Physiological Data Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="paragraph">
            Utilize sophisticated methods to analyze physiological data, including GSR, respiration rate, body temperature, and heart rate, to accurately assess stress levels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Live Stress Detection Section
    st.markdown('<div class="section-header">Live Stress Detection (Emotion Based)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="paragraph">
            Experience real-time stress detection through advanced emotion analysis. Our system monitors and interprets emotional cues to provide immediate feedback on stress levels.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Close main container
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()