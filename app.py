import streamlit as st
import pickle
import numpy as np

# Load model & encoders
model = pickle.load(open("medicine_model.pkl", "rb"))
le_medicine = pickle.load(open("medicine_encoder.pkl", "rb"))
le_disease = pickle.load(open("disease_encoder.pkl", "rb"))

le_safety = pickle.load(open("safety_encoder.pkl", "rb"))

st.set_page_config(page_title="AI Medicine Checker", page_icon="💊")

st.title("💊 AI Medicine Safety Checker")
st.write("Check medicine safety based on age, disease ")

medicine = st.selectbox(
    "Select Medicine",
    le_medicine.classes_
)

age = st.slider("Age", 1, 100, 25)

disease = st.selectbox(
    "Select Disease",
    le_disease.classes_
)



if st.button("Check Safety"):
    med_enc = le_medicine.transform([medicine])[0]
    dis_enc = le_disease.transform([disease])[0]
    

    prediction = model.predict([[med_enc, age, dis_enc, ]])
    result = le_safety.inverse_transform(prediction)[0]

    st.subheader("🔍 Result")

    if result == "Safe":
        st.success("✅ SAFE to use (within normal dosage)")
    elif result == "Caution":
        st.warning("⚠ USE WITH CAUTION – Consult a doctor")
    else:
        st.error("❌ UNSAFE – Do NOT use")

st.markdown("---")
st.caption("⚠ This is an advisory AI system, not medical diagnosis.")
