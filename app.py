import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("model/breast_cancer_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 Breast Cancer Prediction System")
st.write(
    """
    ⚠ **Educational Use Only**  
    This system is strictly for educational purposes and must not be used as a medical diagnostic tool.
    """
)

# Input fields
radius = st.number_input("Radius Mean", 0.0, 30.0, 14.0)
texture = st.number_input("Texture Mean", 0.0, 50.0, 19.0)
perimeter = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0)
area = st.number_input("Area Mean", 0.0, 2500.0, 600.0)
concavity = st.number_input("Concavity Mean", 0.0, 1.0, 0.1)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[radius, texture, perimeter, area, concavity]],
        columns=[
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean concavity"
        ]
    )

    prediction = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction])[0]

    if prediction_label == "Malignant":
        st.error("🔴 Prediction: Malignant")
    else:
        st.success("🟢 Prediction: Benign")
