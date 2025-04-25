import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("heart_disease.csv")

# Data preprocessing (simple example)
X = df.drop("target", axis=1)
y = df["target"]

# Train model (in real app, you might load a pre-trained model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Streamlit App
st.title("Heart Disease Prediction App")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex (1 = male, 0 = female)', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol', 100, 600, 240)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 (1 = true; 0 = false)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG (0-2)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 70, 210, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (1 = yes; 0 = no)', [0, 1])
    oldpeak = st.sidebar.slider('ST depression', 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox('Slope (0-2)', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect)', [1, 2, 3])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display input parameters
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
st.write(result)

st.subheader("Prediction Probability")
st.write(prediction_proba)

st.caption("Note: This is a demo model. For real-world applications, further validation is essential.")
