import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
dataset = pd.read_csv("heart.csv")

# Split data into features and target
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Define Random Forest model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, Y_train)

# Make predictions and calculate accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_pred, Y_test)

# Streamlit interface
st.set_page_config(page_title="Heart Disease", page_icon="❤️")
st.title("Heart Disease Prediction")

# Display data overview
if st.checkbox('Show data overview'):
    st.write(dataset.head())

# User inputs for prediction
st.sidebar.header("Enter Patient Data:")
age = st.sidebar.slider("Age", 29, 77, 50)
sex = st.sidebar.selectbox("Sex (1: Male, 0: Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (1: Typical, 2: Atypical, 3: Non-anginal, 4: Asymptomatic)", [1, 2, 3, 4])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=94, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=126, max_value=564, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)", [1, 0])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (0, 1, 2)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=71, max_value=202, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1: Yes, 0: No)", [1, 0])
oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=6.2, value=1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (1, 2, 3)", [1, 2, 3])
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (3: Normal, 6: Fixed Defect, 7: Reversible Defect)", [3, 6, 7])

# Input data for prediction
input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

# Button to start prediction
if st.button("Start Prediction"):
    # Make prediction using the trained Random Forest model
    prediction = model.predict(input_data)

    # Present the result in a more user-friendly way
    if prediction[0] == 1:
        st.markdown(f"### **Warning: You are at risk of having heart disease.**")
        st.write("Based on the input data you have provided, our model has predicted that you may have heart disease. It is highly recommended that you consult a medical professional for a thorough check-up and diagnosis.")
    else:
        st.markdown(f"### **Good news: You are not at risk of having heart disease.**")
        st.write("Based on the input data you have provided, our model has predicted that you do not have heart disease. However, maintaining a healthy lifestyle is still important for overall health.")

    # Show accuracy of the Random Forest model
    st.write(f"Accuracy of Random Forest model: {accuracy * 100:.2f}%")
