import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# 页面配置
st.set_page_config(
    page_title="免疫抑制脓毒症患者凝血功能障碍预测系统",
    page_icon="🩺",
    layout="centered"
)

# 标题与说明
st.title("🩺 免疫抑制脓毒症患者 7 天内凝血功能障碍风险预测系统")
st.markdown("本系统基于 ICU 入科 24h 内的临床指标，利用机器学习（LightGBM 模型）评估患者在 7 天内发生凝血功能障碍的风险。")

@st.cache_resource
def load_data_and_scaler():
    # 读取训练数据以拟合 StandardScaler（确保和训练脚本一致）
    train_data = pd.read_csv("traindata.csv", encoding="GBK")
    features = ['PLT', 'APTT', 'Lac', 'SOFA', 'MCH', 'Elixhauser', 'Urine_Rate', 'RDW', 'WBC', 'PaO2']
    X_train_raw = train_data[features]
    y_train = train_data['Outcome']
    
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    
    # 直接基于当前数据训练一个与脚本参数一致的 LightGBM 模型，确保独立稳定运行
    model = lgb.LGBMClassifier(
        random_state=123,
        learning_rate=0.1,
        num_leaves=50,
        n_estimators=100,
        subsample=0.6,
        colsample_bytree=0.8
    )
    X_scaled = scaler.transform(X_train_raw)
    model.fit(X_scaled, y_train)
    
    return scaler, model

scaler, model = load_data_and_scaler()
features = ['PLT', 'APTT', 'Lac', 'SOFA', 'MCH', 'Elixhauser', 'Urine_Rate', 'RDW', 'WBC', 'PaO2']

# 侧边栏或主界面输入表单
st.sidebar.header("患者临床指标输入 (ICU 24h内)")

def user_input_features():
    plt_val = st.sidebar.number_input('血小板计数 (PLT)', value=150.0, step=1.0)
    aptt_val = st.sidebar.number_input('活化部分凝血活酶时间 (APTT)', value=35.0, step=0.5)
    lac_val = st.sidebar.number_input('乳酸 (Lac)', value=1.5, step=0.1)
    sofa_val = st.sidebar.number_input('SOFA 评分', value=5.0, step=1.0)
    mch_val = st.sidebar.number_input('平均红细胞血红蛋白含量 (MCH)', value=30.0, step=0.5)
    elixhauser_val = st.sidebar.number_input('Elixhauser 合并症指数', value=2.0, step=1.0)
    urine_rate = st.sidebar.number_input('尿量率 (Urine_Rate)', value=1.0, step=0.1)
    rdw_val = st.sidebar.number_input('红细胞分布宽度 (RDW)', value=14.0, step=0.1)
    wbc_val = st.sidebar.number_input('白细胞计数 (WBC)', value=10.0, step=0.5)
    pao2_val = st.sidebar.number_input('动脉血氧分压 (PaO2)', value=85.0, step=1.0)
    
    data = {
        'PLT': plt_val,
        'APTT': aptt_val,
        'Lac': lac_val,
        'SOFA': sofa_val,
        'MCH': mch_val,
        'Elixhauser': elixhauser_val,
        'Urine_Rate': urine_rate,
        'RDW': rdw_val,
        'WBC': wbc_val,
        'PaO2': pao2_val
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 展示输入参数
st.subheader("📋 当前输入的患者临床参数：")
st.dataframe(input_df)

# 预测按钮
if st.button("开始预测风险", type="primary"):
    # 标准化输入数据
    input_scaled = scaler.transform(input_df[features])
    
    # 预测概率
    pred_prob = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("📊 预测结果")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="7天内发生凝血功能障碍概率", value=f"{pred_prob * 100:.2f}%")
        
    with col2:
        if pred_prob >= 0.5:
            st.error("⚠️ 预测结果：高风险")
            st.write("建议密切监测凝血指标，并尽早采取临床干预措施。")
        else:
            st.success("✅ 预测结果：低风险")
            st.write("患者短期内发生凝血功能障碍的风险相对较低，请继续常规监护。")
            
    # 概率进度条展示
    st.progress(float(pred_prob))