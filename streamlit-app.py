import pickle
import pandas as pd
import streamlit as st

# Load the model and encoder
with open('heart_output_knn.pkl', 'rb') as file:
    model, output_encoder = pickle.load(file)

# Function to make predictions
def predict_species(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    x_new = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trtbps': [trtbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalachh': [thalachh],
        'exng': [exng],
        'oldpeak': [oldpeak],
        'slp': [slp],
        'caa': [caa],
        'thall': [thall]
    })

    y_pred_new = model.predict(x_new)
    result = output_encoder.inverse_transform(y_pred_new)
    return result[0]

# Streamlit UI
st.title("Heart Disease Prediction")
st.sidebar.header("User Input")

# Collect user input
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type", min_value=1, max_value=4, value=4)
trtbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalachh = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
exng = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0)
slp = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
caa = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
thall = st.sidebar.number_input("Thalassemia Type", min_value=0, max_value=3, value=2)

# Make prediction and display result
if st.sidebar.button("Predict"):
    prediction = predict_species(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)
    st.write(f"Predicted Specie: {prediction}")
