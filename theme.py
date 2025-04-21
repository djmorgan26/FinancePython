import streamlit as st

def apply_green_theme():
    """
    Apply the green, white, and black themed styling to the Streamlit app (light mode).
    """
    st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: #006400;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            margin-top: 1.2rem;
            color: #006400;
        }
        .stButton>button {
            background-color: #006400;
            color: white;
        }
        .stSelectbox>div>div {
            background-color: #f0fff0;
        }
        .stDataFrame {
            border: 1px solid #006400;
        }
        .css-1d391kg {
            background-color: #f0fff0;
        }
        .sidebar .sidebar-content {
            background-color: #f0fff0;
        }
        .sidebar .sidebar-content .sidebar-section {
            background-color: #f0fff0;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_blue_theme():
    """
    Apply the black and green themed styling to the Streamlit app (dark mode).
    """
    st.markdown("""
    <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 1.5rem;
            color: blue;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: bold;
            margin-top: 1.2rem;
            color: blue;
        }
        .stButton>button {
            background-color: #00ff00;
            color: black;
        }
        .stSelectbox>div>div {
            background-color: #333333;
            color: #ffffff;
        }
        .stDataFrame {
            border: 1px solid #00ff00;
        }
        .css-1d391kg {
            background-color: #333333;
        }
        .sidebar .sidebar-content {
            background-color: #333333;
        }
        .sidebar .sidebar-content .sidebar-section {
            background-color: #333333;
        }
    </style>
    """, unsafe_allow_html=True)