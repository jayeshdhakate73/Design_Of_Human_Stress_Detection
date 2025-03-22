import streamlit as st

# Set page config at the very beginning of the script
st.set_page_config(
    page_title='Human Stress Detector',
    page_icon=':worried:',
    layout='wide',
    initial_sidebar_state='auto'
)

# Import necessary functions from web_functions
from web_functions import load_data

# Import pages
from Tabs import home, detect, edetect  # Ensure all modules exist

# Dictionary for pages
Tabs = {
    "Home": home,
    "Stress Detection": detect,
    "Live Stress Detection ": edetect,
}

# Create a sidebar
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Loading the dataset.
df, x, y = load_data()

# Run the selected page
if page == "Stress Detection":
    Tabs[page].app(df, x, y)
elif page == "Live Stress Detection ":
    Tabs[page].app()  # No dataset required for this
else:
    Tabs[page].app()