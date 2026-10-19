import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ==========================================
# 1. 页面全局配置 (必须是第一行 Streamlit 命令)
# ==========================================
st.set_page_config(
    page_title="SIC Risk Calculator",
    page_icon="⚕️",
    layout="wide" # 强制宽屏模式，支持左右分栏
)

# 自定义 CSS 美化
st.markdown("""
    <style>
    .main {background-color: #FAFAFA;}
    h1, h2, h3, h4 {color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif;}
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
# 2. 顶部标题与背景介绍
# ==========================================
st.markdown("<h1>Dynamic Online Calculator for SIC Risk</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <b>Objective:</b> To predict the 7-day risk of Sepsis-Induced Coagulopathy (SIC) in immunosuppressed patients with sepsis based on clinical data obtained within 24 hours of ICU admission.<br>
    <b>Development & Validation:</b> The machine learning model (LightGBM) was trained using the MIMIC database and externally validated with data from the First Medical Center of Chinese PLA General Hospital.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('lgb_model.pkl')
    except Exception as e:
        return None

model = load_model()
st.markdown("---")

# ==========================================
# 3. 左右分栏核心布局 (左侧填表，右侧出图)
# ==========================================
left_col, right_col = st.columns([1.2, 1], gap="large")

# ----------------- 左侧：输入区域 -----------------
with left_col:
    st.markdown("### Patient Clinical Profiles")
    st.markdown("*(Within 24h of ICU Admission)*")
    
    in_col1, in_col2 = st.columns(2)
    
    with in_col1:
        st.markdown("#### 🩸 Hemogram & Coagulation")
        plt = st.number_input("Platelets (PLT) [×10^9/L]", min_value=0.0, max_value=1000.0, value=150.0)
        aptt = st.number_input("Activated Partial Thromboplastin Time (APTT) [s]", min_value=0.0, max_value=200.0, value=35.0)
        mch = st.number_input("Mean Corpuscular Hemoglobin (MCH) [pg]", min_value=0.0, max_value=60.0, value=29.0)
        rdw = st.number_input("Red Cell Distribution Width (RDW) [%]", min_value=0.0, max_value=40.0, value=13.0)
        wbc = st.number_input("White Blood Cell (WBC) [×10^9/L]", min_value=0.0, max_value=200.0, value=10.0)

    with in_col2:
        st.markdown("#### 🫀 Organ Function & Metabolic")
        sofa = st.number_input("SOFA Score", min_value=0, max_value=24, value=5)
        lac = st.number_input("Lactate (Lac) [mmol/L]", min_value=0.0, max_value=30.0, value=2.0)
        pao2 = st.number_input("PₐO₂ [mmHg]", min_value=0.0, max_value=500.0, value=80.0)
        # 修正了尿量单位和范围
        urine_rate = st.number_input("Urine Output Rate [mL/h]", min_value=0.0, max_value=500.0, value=70.0)
        
    st.markdown("#### 🏥 Comorbidities")
    elixhauser = st.slider("Elixhauser Comorbidity Index", min_value=-20, max_value=50, value=0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict SIC Risk")

# ----------------- 右侧：结果区域 -----------------
with right_col:
    st.markdown("### Prediction Result")
    
    if predict_btn:
        if model is not None:
            user_data = {
                'PLT': plt, 'APTT': aptt, 'Lac': lac, 'SOFA': sofa, 'MCH': mch, 
                'Elixhauser': elixhauser, 'Urine_Rate': urine_rate, 'RDW': rdw, 'WBC': wbc, 'PaO2': pao2
            }
            input_df = pd.DataFrame(user_data, index=[0])
            
            try:
                with st.spinner('Standardizing features and calculating risk probability...'):
                    # 动态读取训练集用于 Z-score 标准化
                    train_data = pd.read_csv('traindata.csv')
                    features_order = ['PLT', 'APTT', 'Lac', 'SOFA', 'MCH', 'Elixhauser', 'Urine_Rate', 'RDW', 'WBC', 'PaO2']
                    
                    train_means = train_data[features_order].mean()
                    train_stds = train_data[features_order].std()
                    
                    # 严格按照训练集的均值和标准差缩放输入
                    scaled_input_df = (input_df[features_order] - train_means) / train_stds
                    
                    # 模型预测
                    proba = model.predict_proba(scaled_input_df)[0][1]
                    proba_percentage = proba * 100
                    
                    # 绘制高级仪表盘
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = proba_percentage,
                        number = {'suffix': "%", 'font': {'size': 45, 'color': '#1E3A8A'}},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability of SIC within 7 Days", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "rgba(0,0,0,0)"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#86EFAC'},
                                {'range': [30, 60], 'color': '#FDE047'},
                                {'range': [60, 100], 'color': '#FCA5A5'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 6},
                                'thickness': 0.75,
                                'value': proba_percentage
                            }
                        }
                    ))
                    
                    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 临床指导建议
                    if proba_percentage >= 60:
                        st.error("⚠️ **High Risk Profile:** The patient demonstrates a high probability of developing Sepsis-Induced Coagulopathy. Close monitoring of coagulation profiles and early interventions are highly recommended.")
                    elif proba_percentage >= 30:
                        st.warning("⚠️ **Moderate Risk Profile:** The patient is at moderate risk. Routine clinical vigilance is advised.")
                    else:
                        st.success("✅ **Low Risk Profile:** The patient currently shows a low probability of developing SIC.")
                        
            except FileNotFoundError:
                st.error("Error: 'traindata.csv' not found. Data standardization requires the original training dataset.")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {str(e)}")
        else:
            st.error("Error: 'lgb_model.pkl' not found. Please ensure the model file is uploaded in the root directory.")
    else:
        st.info("👈 Please enter the patient's clinical parameters on the left and click **'Predict SIC Risk'** to generate the risk assessment and gauge chart.")

# ==========================================
# 4. 底部免责声明
# ==========================================
st.markdown("""
<div class="disclaimer">
    <b>Disclaimer:</b> This predictive model is intended strictly for academic and scientific research purposes. It does not replace clinical judgment. Healthcare providers should interpret the results in the context of the individual patient's comprehensive clinical status.<br>
    <b>Reference:</b> [Insert Your Citation Here]<br>
    <i>© 2024 Chinese PLA General Hospital & Collaborators. All rights reserved.</i>
</div>
""", unsafe_allow_html=True)