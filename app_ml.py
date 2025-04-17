import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64

# Set page configuration
st.set_page_config(
    page_title="AgriPredict - Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    div[data-testid="stVerticalBlock"] {
        padding: 10px;
    }
    div[data-testid="stHeader"] {
        background-color: rgba(18, 18, 18, 0.8);
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 5px;
        padding: 12px 24px;
        border: none;
        font-weight: 600;
        font-size: 18px !important;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #218838;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    h1 {
        color: #ffffff;
        font-size: 3rem !important;
    }
    h2 {
        color: #ffffff;
        font-size: 2.2rem !important;
    }
    h3 {
        color: #ffffff;
        font-size: 1.8rem !important;
    }
    p, div, label, .stMarkdown {
        font-size: 1.2rem !important;
        color: #ffffff;
    }
    .stNumberInput label, .stSlider label {
        font-size: 1.2rem !important;
        color: #ffffff !important;
    }
    .css-10trblm {
        color: #ffffff;
    }
    /* Card styling */
    .card {
        border-radius: 15px;
        padding: 25px;
        background-color: #1e2130;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin-bottom: 25px;
    }
    /* Welcome text style */
    .welcome-text {
        color: #ffffff;
        font-size: 1.4rem !important;
    }
    /* Result card styling */
    .result-card {
        border-radius: 15px;
        padding: 30px 20px;
        background-color: #1e2130;
        border-left: 5px solid #7cb342;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .result-crop {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #7cb342 !important;
        margin: 0;
        line-height: 1.2;
    }
    .result-text {
        font-size: 1.4rem !important;
        color: #ffffff !important;
        margin-top: 5px;
    }
    .metric-card {
        background-color: #1e2130;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    /* Override Streamlit's default text color in widgets */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        color: white !important;
        font-size: 1.2rem !important;
    }
    /* Progress bar colors */
    .stProgress > div > div > div {
        background-color: #7cb342 !important;
    }
    /* Input label colors */
    div[data-baseweb="base-input"] {
        color: white !important;
    }
    label {
        color: white !important;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.3rem !important;
        color: white !important;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #999999;
        font-size: 1rem !important;
    }
    /* Center parameter help texts */
    .stMarkdown div[data-testid="stText"] small {
        color: #aaaaaa !important;
    }
    /* Range info styling */
    .range-info {
        font-size: 0.9rem !important;
        color: #aaaaaa !important;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    /* Info box styling */
    .info-box {
        background-color: #1e2130;
        border: 1px solid #2e3445;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    /* Parameter range details */
    .param-details {
        background-color: rgba(40, 167, 69, 0.1);
        border-left: 3px solid #28a745;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model & preprocessors
@st.cache_resource
def load_ml_resources():
    model = joblib.load("final_rf_crop_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, label_encoder, scaler

model, label_encoder, scaler = load_ml_resources()

# Header section
st.markdown("<h1 style='text-align: center; margin-bottom: 30px;'>🌱 AgriPredict: Smart Crop Recommendation</h1>", unsafe_allow_html=True)

# Introduction
with st.container():
    st.markdown("""
    <div class="card">
        <h3>Welcome to AgriPredict</h3>
        <p class="welcome-text">This intelligent system analyzes soil conditions and environmental factors to recommend the most suitable crop for your land. 
        Our machine learning model is trained on extensive agricultural data to provide accurate recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

# Main content in two columns
col1, col2 = st.columns([3, 2])

# Input Parameters Column
with col1:
    st.markdown("<h3>Soil & Environmental Parameters</h3>", unsafe_allow_html=True)
    
    # Parameter details box
    st.markdown("""
    <div class="param-details">
        <p>Please enter values within the recommended ranges for accurate predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns(2)
    
    with input_col1:
        N = st.number_input("Nitrogen (N) content in soil", 
                            min_value=0, max_value=200, step=1, value=50,
                            help="Amount of nitrogen in kg/ha")
        st.markdown("<div class='range-info'>Range: 0-200 kg/ha (Optimal: 80-120)</div>", unsafe_allow_html=True)
        
        P = st.number_input("Phosphorus (P) content in soil", 
                            min_value=0, max_value=200, step=1, value=50,
                            help="Amount of phosphorus in kg/ha")
        st.markdown("<div class='range-info'>Range: 0-200 kg/ha (Optimal: 40-80)</div>", unsafe_allow_html=True)
        
        K = st.number_input("Potassium (K) content in soil", 
                            min_value=0, max_value=200, step=1, value=50,
                            help="Amount of potassium in kg/ha")
        st.markdown("<div class='range-info'>Range: 0-200 kg/ha (Optimal: 40-80)</div>", unsafe_allow_html=True)
    
    with input_col2:
        pH = st.slider("Soil pH Level", 
                      min_value=3.0, max_value=9.0, step=0.1, value=6.5,
                      help="pH level of the soil (7 is neutral)")
        st.markdown("<div class='range-info'>Range: 3.0-9.0 (Neutral: 6.5-7.5)</div>", unsafe_allow_html=True)
        
        rainfall = st.slider("Annual Rainfall (mm)", 
                           min_value=0.0, max_value=2000.0, step=10.0, value=500.0,
                           help="Average annual rainfall in millimeters")
        st.markdown("<div class='range-info'>Range: 0-2000 mm/year</div>", unsafe_allow_html=True)
        
        temperature = st.slider("Average Temperature (°C)", 
                              min_value=0.0, max_value=50.0, step=0.5, value=25.0,
                              help="Average temperature in degrees Celsius")
        st.markdown("<div class='range-info'>Range: 0-50°C (Most crops: 15-30°C)</div>", unsafe_allow_html=True)

    # Submit Button with spacing
    st.write("")
    predict_btn = st.button("Predict Best Crop 🔍", use_container_width=True)

# Information and Results Column
with col2:
    st.markdown("<h3>Recommendation Results</h3>", unsafe_allow_html=True)
    
    results_placeholder = st.empty()
    
    if predict_btn:
        with st.spinner("Analyzing soil parameters..."):
            # Prepare input
            input_features = np.array([[N, P, K, pH, rainfall, temperature]])
            input_features_scaled = scaler.transform(input_features)
            
            # Predict
            predicted_class = model.predict(input_features_scaled)[0]
            predicted_crop = label_encoder.inverse_transform([predicted_class])[0]
            
            # Get prediction probabilities for visualization
            probabilities = model.predict_proba(input_features_scaled)[0]
            classes = model.classes_
            top_3_indices = probabilities.argsort()[-3:][::-1]
            top_3_crops = label_encoder.inverse_transform(classes[top_3_indices])
            top_3_probs = probabilities[top_3_indices] * 100  # Convert to percentage
            
            # Display results
            with results_placeholder.container():
                st.markdown(f"""
                <div class="result-card">
                    <h2 class="result-crop">🌾 {predicted_crop.title()}</h2>
                    <p class="result-text">is the recommended crop for your soil conditions</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display soil health metrics
                st.markdown("<h3 style='margin-top: 25px;'>Your Soil Profile</h3>", unsafe_allow_html=True)
                
                metric_cols = st.columns(3)
                
                # Convert soil parameters to easy-to-understand ratings
                ph_status = "Acidic" if pH < 6.5 else "Neutral" if pH < 7.5 else "Alkaline"
                n_status = "Low" if N < 50 else "Medium" if N < 100 else "High"
                rain_status = "Low" if rainfall < 500 else "Medium" if rainfall < 1000 else "High"
                
                metric_cols[0].metric("Soil pH", f"{pH:.1f}", ph_status)
                metric_cols[1].metric("Nitrogen Level", f"{N} kg/ha", n_status)
                metric_cols[2].metric("Rainfall", f"{rainfall:.0f} mm", rain_status)
                
                # Display top crop recommendations
                st.markdown("<h3 style='margin-top: 25px;'>Top Crop Recommendations</h3>", unsafe_allow_html=True)
                
                # Create a custom chart using HTML/CSS instead of matplotlib
                for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs)):
                    col_a, col_b = st.columns([1, 4])
                    with col_a:
                        st.markdown(f"<div style='font-size:1.3rem; font-weight:bold;'>{crop.title()}</div>", unsafe_allow_html=True)
                    with col_b:
                        # Create progress bar with percentage
                        st.progress(prob/100)
                        st.markdown(f"<div style='text-align: right; margin-top: -20px; font-size: 1.1rem;'>{prob:.1f}%</div>", unsafe_allow_html=True)
    else:
        # Show instructions when no prediction has been made
        with results_placeholder.container():
            st.info("👈 Enter your soil parameters and click 'Predict Best Crop' to get recommendations.")
            st.write("The model will analyze your inputs and suggest the most suitable crop for your agricultural conditions.")
            
            # Simple text explanation instead of SVG
            st.markdown("""
            <div class="info-box" style="text-align: center; margin-top: 20px; margin-bottom: 20px; padding: 20px;">
                <h3 style="color: #7cb342;">How Crop Recommendations Work</h3>
                <p style="margin-top: 15px;">
                    <b>Step 1:</b> Your soil inputs (N, P, K, pH, rainfall, temperature) are collected<br><br>
                    <b>Step 2:</b> Our Random Forest machine learning model analyzes the data<br><br>
                    <b>Step 3:</b> The model produces crop recommendations with confidence scores
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='text-align: center;'>Based on your soil parameters, we'll recommend crops that will thrive in your conditions</p>", unsafe_allow_html=True)

# Additional info section with helpful resources
st.markdown("---")
info_col1, info_col2 = st.columns(2)

with info_col1:
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        This application uses a Random Forest machine learning model trained on agricultural data to recommend optimal crops based on:
        
        - **N, P, K values**: The concentration of Nitrogen, Phosphorus, and Potassium in the soil
        - **pH**: The acidity or alkalinity of the soil
        - **Rainfall**: Average annual rainfall in millimeters
        - **Temperature**: Average temperature in degrees Celsius
        
        The model evaluates these parameters against historical crop performance to make its recommendation.
        """)

with info_col2:
    with st.expander("📊 About the Dataset"):
        st.markdown("""
        The model was trained using a comprehensive agricultural dataset containing soil quality metrics, environmental conditions, and successful crop yields across various regions.
        
        The data includes information about:
        - Various soil properties across different geographical locations
        - Climate conditions including rainfall and temperature
        - Successful crop yields in different environments
        """)

# Footer
st.markdown("""
<div class="footer">
    <p>AgriPredict - Smart Crop Recommendation System</p>
</div>
""", unsafe_allow_html=True)
