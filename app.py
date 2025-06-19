import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os

# Set the directory for model and data files
MODEL_DIR = 'train_model'

# Load model and features
model = joblib.load(os.path.join(MODEL_DIR, 'heart_model_pipeline.joblib'))
all_features = joblib.load(os.path.join(MODEL_DIR, 'model_features.joblib'))
top_features = ['cp', 'ca', 'thal', 'thalach', 'oldpeak']

# Load training data to get medians/modes for autofill
train_df = pd.read_csv(os.path.join(MODEL_DIR, 'heart.csv'))
medians = train_df.median(numeric_only=True)
modes = train_df.mode().iloc[0]

def get_default_value(col):
    if col in medians:
        return medians[col]
    return modes[col]

st.set_page_config(page_title="Heart Disease", page_icon="❤️")
st.title("Heart Disease Risk Checker")
st.write("""
This tool predicts your risk of heart disease using only the most important information. 
Please enter the following details about yourself:
""")

st.sidebar.header("Enter Your Information:")
cp = st.sidebar.selectbox(
    "Type of Chest Pain",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical angina (chest pain from heart)",
        2: "Atypical angina (unusual chest pain)",
        3: "Non-anginal pain (not heart-related)",
        4: "No chest pain (asymptomatic)"
    }[x],
    help="What best describes your chest pain?"
)
ca = st.sidebar.selectbox(
    "Number of Major Heart Vessels Blocked (0-3)",
    [0, 1, 2, 3],
    help="How many major blood vessels are blocked, as seen in a scan? (0 means none)"
)
thal = st.sidebar.selectbox(
    "Thalassemia Type (a blood disorder)",
    [3, 6, 7],
    format_func=lambda x: {
        3: "Normal blood",
        6: "Fixed defect (old heart damage)",
        7: "Reversible defect (current heart issue)"
    }[x],
    help="Has a doctor told you about a blood disorder or heart scan result?"
)
thalach = st.sidebar.number_input(
    "Maximum Heart Rate Achieved (bpm)",
    min_value=71, max_value=202, value=150,
    step=1,
    help="What was the highest your heart rate got during exercise or a test? (beats per minute)"
)
oldpeak = st.sidebar.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0, max_value=6.2, value=1.0,
    step=0.1,
    format="%.2f",
    help="How much did your ECG reading drop during exercise? (Ask your doctor if unsure)"
)

# Build input row for the model using only the 5 features
input_data = pd.DataFrame([[cp, ca, thal, thalach, oldpeak]], columns=top_features)

if st.button("Check My Heart Disease Risk"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    if prediction == 1:
        st.markdown(f"## ✅ You are likely not at risk of heart disease.")
        st.write(f"**Estimated risk:** {1-proba:.2%}")
        st.success("Keep maintaining a healthy lifestyle!")
    else:
        st.markdown(f"## ⚠️ You may be at risk of heart disease.")
        st.write(f"**Estimated risk:** {1-proba:.2%}")
        st.info("This is not a diagnosis. Please consult a doctor for a full check-up.")