import streamlit as st
import numpy as np
import plotly.graph_objects as go  # For enhanced visualizations

from web_functions import detect

def app(df, x, y):

    # Custom CSS for a sleek, modern look
    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .stSlider > div > div > div > div {
            background-color: #4CAF50; /* Green theme for sliders */
        }
        .stButton > button {
            background-color: #008CBA; /* Blue button */
            color: white;
            padding: 14px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #0077A3; /* Darker blue on hover */
        }
        .stInfo, .stSuccess, .stWarning, .stError {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .stInfo { background-color: #e0f7fa; border: 1px solid #b2ebf2; color: #006064; }
        .stSuccess { background-color: #e8f5e9; border: 1px solid #c8e6c9; color: #2e7d32; }
        .stWarning { background-color: #fffde7; border: 1px solid #fff9c4; color: #f9a825; }
        .stError { background-color: #ffebee; border: 1px solid #ef9a9a; color: #d32f2f; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🩺 Advanced Stress Level Detection System")

    st.markdown(
        """
        <p style="font-size:20px;">
            Utilizing a sophisticated <b style="color:#008CBA;">Decision Tree Classifier</b>, this system provides precise stress level assessments based on physiological data.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("📊 Physiological Data Input")
    col1, col2 = st.columns(2)

    with col1:
        gsr = st.slider("Galvanic Skin Response (GSR)", int(df["gsr"].min()), int(df["gsr"].max()))
        rr = st.slider("Respiration Rate", int(df["rr"].min()), int(df["rr"].max()))
        bt = st.slider("Body Temperature (°F)", int(df["bt"].min()), int(df["bt"].max()))
        lm = st.slider("Limb Movement", float(df["lm"].min()), float(df["lm"].max()))

    with col2:
        bo = st.slider("Blood Oxygen (%)", float(df["bo"].min()), float(df["bo"].max()))
        rem = st.slider("Rapid Eye Movement", float(df["rem"].min()), float(df["rem"].max()))
        sh = st.slider("Sleeping Hours", float(df["sh"].min()), float(df["sh"].max()))
        hr = st.slider("Heart Rate", float(df["hr"].min()), float(df["hr"].max()))

    features = [gsr, rr, bt, lm, bo, rem, sh, hr]

    if st.button("Analyze Stress Level"):
        detection, score = detect(x, y, features)
        st.info("Analyzing physiological data...")

        if detection == 1:
            st.success("Stress Level: Low 🙂")
        elif detection == 2:
            st.warning("Stress Level: Medium 😐")
        elif detection == 3:
            st.error("Stress Level: High 😞")
        elif detection == 4:
            st.error("Stress Level: Very High 😫")
        else:
            st.success("Stress Level: Calm and Stress-Free 😄")

        # Visualization of input data
        fig = go.Figure(data=[go.Bar(
            x=["GSR", "RR", "BT", "LM", "BO", "REM", "SH", "HR"],
            y=features,
            marker_color=['#008CBA', '#4CAF50', '#FFC107', '#E91E63', '#9C27B0', '#673AB7', '#3F51B5', '#2196F3']
        )])
        fig.update_layout(title="Physiological Data Input", xaxis_title="Metrics", yaxis_title="Values")
        st.plotly_chart(fig)

        # Display the score
        st.write(f"Model Confidence Score: {score:.2f}")

    st.markdown("---")
    st.markdown(
        """
        <p style="font-size:14px;">
            Disclaimer: This system provides an estimated stress level based on input data. For medical advice, consult a healthcare professional.
        </p>
        """,
        unsafe_allow_html=True,
    )