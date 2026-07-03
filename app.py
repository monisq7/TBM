import streamlit as st
import pandas as pd
import pickle
from database import save_prediction

# ---------- Model ----------
with open("mortality_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="TBM Mortality Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------- CSS ----------
st.markdown("""
<style>
.main {
    background: #f4f7fc;
}
.title-box {
    background: linear-gradient(135deg, #0f4c81, #2a9d8f);
    padding: 25px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
div.stButton > button {
    width: 100%;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="title-box">
    <h1>🏥 TBM Mortality Prediction System</h1>
    <p>Random Forest Based Clinical Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About Model")
st.sidebar.info("""
Model: Random Forest  
Trees: 35  
Target: In-hospital Mortality  
""")

st.sidebar.metric("Model Accuracy", "91.3%")
st.sidebar.metric("AUC Score", "0.94")

# Patient Name (not used by the model, only stored alongside the prediction)
patient_name = st.text_input("Patient Name")

# Inputs
tab1, tab2 = st.tabs(["Basic Parameters", "Clinical Parameters"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 0.0, 120.0, 39.0)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        fever = st.number_input("Duration of Fever (Months)", value=6.0)
        headache = st.number_input("Duration of Headache (Months)", value=6.0)
        sensorium = st.toggle("Altered Sensorium")

    with col2:
        neuro = st.toggle("Focal Neurological Deficit")
        gcs = st.slider("GCS", 3, 15, 12)
        icp = st.toggle("Raised ICP")
        sodium = st.number_input("Sodium Na (mEQ/L)", value=135.0)

with tab2:
    col3, col4 = st.columns(2)

    with col3:
        csf = st.number_input("CSF Count", value=96.0)
        protein = st.number_input("Protein (mg/dL)", value=600.0)
        glucose = st.number_input("Glucose (mg/dL)", value=52.0)

    with col4:
        hydro = st.toggle("Hydrocephalus")
        exudates = st.toggle("Basal Exudates")
        infarcts = st.toggle("Infarcts")

if st.button("Predict Mortality Risk"):

    input_data = pd.DataFrame([[
        age,
        sex,
        fever,
        headache,
        int(sensorium),
        int(neuro),
        gcs,
        int(icp),
        sodium,
        csf,
        protein,
        glucose,
        int(hydro),
        int(exudates),
        int(infarcts)
    ]], columns=[
        "Age (years)",
        "Sex",
        "Duration of fever (Months)",
        "Duration of headache (Months)",
        "Altered sensorium (Y/N)",
        "Focal neurological deficit (Ophthalmoplegia/Hemiplegia/Other)",
        "GCS (E+V+M out of 15)",
        "Raised ICP",
        "Na (mEQ/L)",
        "CSF Count (cells/µL)",
        "Protein (mg/dL)",
        "Glucose (mg/dL)",
        "Hydrocephalus (Y/N)",
        "Basal exudates (Y/N)",
        "Infarcts (Y/N)"
    ])

    # Prediction
    prob = model.predict_proba(input_data)[0]
    death_prob = prob[1] * 100

    st.subheader("🩺 Prediction Result")

    # Risk category
    if death_prob <= 25:
        risk = "LOW RISK"
        color = "#2ecc71"
        icon = "✅"
        recommendation = "Patient is clinically stable. Continue routine monitoring."

    elif death_prob <= 50:
        risk = "MEDIUM RISK"
        color = "#f39c12"
        icon = "⚠️"
        recommendation = "Patient requires close observation and frequent reassessment."

    else:
        risk = "HIGH RISK"
        color = "#e74c3c"
        icon = "🚨"
        recommendation = "Critical condition. Immediate intervention and ICU monitoring recommended."

    # Result Card
    st.markdown(f"""
    <div style="
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        border-left: 12px solid {color};
        margin-top:20px;
    ">
        <h1 style="color:{color}; margin-bottom:10px;">
            {icon} {risk}
        </h1>
        <h3>Clinical Recommendation</h3>
        <p style="font-size:18px;">
            {recommendation}
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Patient Summary"):
        st.write(f"**Patient Name:** {patient_name.strip() if patient_name else 'Unnamed'}")
        st.write(input_data)

    # ---------- Save to MongoDB ----------
    saved, error = save_prediction(input_data, death_prob, risk, patient_name)
    if saved:
        st.toast("✅ Prediction saved to database", icon="💾")
    else:
        st.warning(f"⚠️ Could not save to database: {error}")

# Footer
st.markdown("---")
st.caption("TBM Mortality Prediction System | Random Forest Model | IET Lucknow Research Project")
