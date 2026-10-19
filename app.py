import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# 1. Page Configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="SAC Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Header and Description
st.title("Prediction of Sepsis-Associated Coagulopathy (SAC)")
st.markdown("""
**Development and validation of a machine learning model for predicting the 7-day risk of SAC in immunosuppressed sepsis patients.**
This tool utilizes clinical data obtained within 24 hours of ICU admission to stratify patient risk, facilitating early clinical decision-making.
""")
st.markdown("---")

# 3. Sidebar for Patient Clinical Parameters
st.sidebar.header("Patient Clinical Parameters")
st.sidebar.markdown("*(24h post ICU admission)*")

def get_user_input():
    # Arranging features in the exact order required by the LightGBM model
    plt = st.sidebar.number_input("PLT (×10^9/L)", min_value=0.0, max_value=1000.0, value=150.0, step=1.0)
    aptt = st.sidebar.number_input("APTT (s)", min_value=0.0, max_value=200.0, value=35.0, step=0.1)
    lac = st.sidebar.number_input("Lactate (mmol/L)", min_value=0.0, max_value=30.0, value=2.0, step=0.1)
    sofa = st.sidebar.number_input("SOFA Score", min_value=0, max_value=24, value=5, step=1)
    mch = st.sidebar.number_input("MCH (pg)", min_value=0.0, max_value=60.0, value=29.0, step=0.1)
    elixhauser = st.sidebar.number_input("Elixhauser Comorbidity Index", min_value=-20, max_value=50, value=0, step=1)
    urine_rate = st.sidebar.number_input("Urine Output Rate (mL/kg/h)", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    rdw = st.sidebar.number_input("RDW (%)", min_value=0.0, max_value=40.0, value=13.0, step=0.1)
    wbc = st.sidebar.number_input("WBC (×10^9/L)", min_value=0.0, max_value=200.0, value=10.0, step=0.1)
    pao2 = st.sidebar.number_input("PaO2 (mmHg)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)

    user_data = {
        'PLT': plt,
        'APTT': aptt,
        'Lac': lac,
        'SOFA': sofa,
        'MCH': mch,
        'Elixhauser': elixhauser,
        'Urine_Rate': urine_rate,
        'RDW': rdw,
        'WBC': wbc,
        'PaO2': pao2
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

input_df = get_user_input()

# 4. Main Panel: Display Inputs
st.subheader("1. Clinical Data Summary")
st.dataframe(input_df, hide_index=True)

# 5. Load the Trained Model
@st.cache_resource
def load_model():
    # Replace 'lgbm_model.pkl' with the actual path to your serialized LightGBM model
    # e.g., model = joblib.load('lgbm_model.pkl')
    # For demonstration, we assume the model is saved as lgbm_model.pkl
    try:
        model = joblib.load('lgbm_model.pkl')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# 6. Prediction Section
st.subheader("2. Risk Prediction")

if st.button("Calculate Probability", type="primary"):
    if model is not None:
        # Get probability of the positive class (assuming index 1 is SAC positive)
        proba = model.predict_proba(input_df)[0][1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Probability of SAC within 7 Days", value=f"{proba:.2%}")
        
        with col2:
            if proba > 0.5: # You can adjust the threshold based on your Youden's index / ROC curve
                st.error("⚠️ **High Risk**")
                st.markdown("This patient is at a high risk of developing Sepsis-Associated Coagulopathy.")
            else:
                st.success("✅ **Low Risk**")
                st.markdown("This patient is at a low risk of developing Sepsis-Associated Coagulopathy.")
        
        # Display a progress bar for visual impact
        st.progress(float(proba))
        
    else:
        st.warning("Model file not found. Please ensure 'lgbm_model.pkl' is uploaded in the root directory.")

# 7. Footer / Disclaimer
st.markdown("---")
st.markdown("""
### Disclaimer
*This web-based tool is designed strictly for academic research and validation purposes. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Clinical decisions must be made by qualified healthcare providers.*
""")