import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration & Custom CSS
# ==========================================
st.set_page_config(
    page_title="SIC Risk Calculator",
    page_icon="⚕️",
    layout="centered" # Using centered layout for a more focused, document-like feel
)

# Custom CSS for academic, clean look
st.markdown("""
    <style>
    .main {background-color: #FAFAFA;}
    h1, h2, h3 {color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif;}
    .stButton>button {
        background-color: #1E3A8A; color: white; width: 100%; border-radius: 5px; 
        font-weight: bold; padding: 10px; border: none; transition: 0.3s;
    }
    .stButton>button:hover {background-color: #2563EB; color: white; border: none;}
    .info-box {
        background-color: #E0F2FE; padding: 15px; border-left: 5px solid #0284C7; 
        border-radius: 5px; margin-bottom: 20px; font-size: 14px; color: #333;
    }
    .disclaimer {font-size: 12px; color: #6B7280; margin-top: 50px; border-top: 1px solid #E5E7EB; padding-top: 20px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Header and Academic Background
# ==========================================
st.markdown("<h1>Dynamic Online Calculator for SIC Risk</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <b>Objective:</b> To predict the 7-day risk of Sepsis-Induced Coagulopathy (SIC) in immunosuppressed patients with sepsis based on clinical data obtained within 24 hours of ICU admission.<br>
    <b>Development & Validation:</b> The machine learning model (LightGBM) was trained using the MIMIC database and externally validated with data from the First Medical Center of Chinese PLA General Hospital.
</div>
""", unsafe_allow_html=True)

st.markdown("### Patient Clinical Profiles (Within 24h of ICU Admission)")

# ==========================================
# 3. Clinical Inputs (Grouped logically)
# ==========================================
# Using columns and expanders to create a clinical form layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🩸 Hemogram & Coagulation")
    plt = st.number_input("Platelets (PLT) [×10^9/L]", min_value=0.0, max_value=1000.0, value=150.0)
    aptt = st.number_input("Activated Partial Thromboplastin Time (APTT) [s]", min_value=0.0, max_value=200.0, value=35.0)
    mch = st.number_input("Mean Corpuscular Hemoglobin (MCH) [pg]", min_value=0.0, max_value=60.0, value=29.0)
    rdw = st.number_input("Red Cell Distribution Width (RDW) [%]", min_value=0.0, max_value=40.0, value=13.0)
    wbc = st.number_input("White Blood Cell (WBC) [×10^9/L]", min_value=0.0, max_value=200.0, value=10.0)

with col2:
    st.markdown("#### 🫀 Organ Function & Metabolic")
    sofa = st.number_input("SOFA Score", min_value=0, max_value=24, value=5)
    lac = st.number_input("Lactate (Lac) [mmol/L]", min_value=0.0, max_value=30.0, value=2.0)
    pao2 = st.number_input("PaO2 [mmHg]", min_value=0.0, max_value=500.0, value=80.0)
    urine_rate = st.number_input("Urine Output Rate [mL/kg/h]", min_value=0.0, max_value=20.0, value=1.0)
    
st.markdown("#### 🏥 Comorbidities")
elixhauser = st.slider("Elixhauser Comorbidity Index", min_value=-20, max_value=50, value=0)

# Create DataFrame maintaining exact order from lightgbm model
user_data = {
    'PLT': plt, 'APTT': aptt, 'Lac': lac, 'SOFA': sofa, 'MCH': mch, 
    'Elixhauser': elixhauser, 'Urine_Rate': urine_rate, 'RDW': rdw, 'WBC': wbc, 'PaO2': pao2
}
input_df = pd.DataFrame(user_data, index=[0])

# ==========================================
# 4. Model Loading & Prediction
# ==========================================
st.markdown("---")

@st.cache_resource
def load_model():
    try:
        # Load the specified model file
        model = joblib.load('lgb_model.pkl')
        return model
    except Exception as e:
        return None

model = load_model()

# ==========================================
# 5. Result Visualization (Gauge Chart)
# ==========================================
if st.button("Predict SIC Risk"):
    if model is not None:
        with st.spinner('Calculating risk probability...'):
            # Predict probability
            proba = model.predict_proba(input_df)[0][1]
            proba_percentage = proba * 100
            
            st.markdown("### Prediction Result")
            
            # Create a professional Gauge Chart using Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = proba_percentage,
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#1E3A8A'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of SIC within 7 Days", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)"}, # Hide the default bar, use threshold
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#86EFAC'},     # Green / Low Risk
                        {'range': [30, 60], 'color': '#FDE047'},    # Yellow / Moderate Risk
                        {'range': [60, 100], 'color': '#FCA5A5'}    # Red / High Risk
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 5},
                        'thickness': 0.75,
                        'value': proba_percentage
                    }
                }
            ))
            
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical interpretation
            if proba_percentage >= 60:
                st.error("⚠️ **High Risk Profile:** The patient demonstrates a high probability of developing Sepsis-Induced Coagulopathy. Close monitoring of coagulation profiles and early interventions are highly recommended.")
            elif proba_percentage >= 30:
                st.warning("⚠️ **Moderate Risk Profile:** The patient is at moderate risk. Routine clinical vigilance is advised.")
            else:
                st.success("✅ **Low Risk Profile:** The patient currently shows a low probability of developing SIC.")
                
    else:
        st.error("Error: 'lgb_model.pkl' not found. Please ensure the model file is uploaded in the root directory.")

# ==========================================
# 6. Academic Footer
# ==========================================
st.markdown("""
<div class="disclaimer">
    <b>Disclaimer:</b> This predictive model is intended strictly for academic and scientific research purposes. It does not replace clinical judgment. Healthcare providers should interpret the results in the context of the individual patient's comprehensive clinical status.<br>
    <b>Reference:</b> [Insert Your Citation Here, e.g., Author A, et al. Machine learning for predicting SIC... Intensive Care Med. 2024.]<br>
    <i>© 2024 Chinese PLA General Hospital & Collaborators. All rights reserved.</i>
</div>
""", unsafe_allow_html=True)