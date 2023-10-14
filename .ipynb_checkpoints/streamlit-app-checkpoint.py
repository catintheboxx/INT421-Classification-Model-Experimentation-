import streamlit as st
import pickle
import pandas as pd

# Load the model and encoder
with open('heart_output_knn.pkl', 'rb') as file:
    model, output_encoder = pickle.load(file)

# Define the function for making predictions
def predict_species(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    x_new = pd.DataFrame({'age': [age], 'sex': [sex], 'cp': [cp], 'trtbps': [trtbps], 'chol': [chol], 'fbs': [fbs],
                          'restecg': [restecg], 'thalachh': [thalachh], 'exng': [exng], 'oldpeak': [oldpeak],
                          'slp': [slp], 'caa': [caa], 'thall': [thall], 'output': [5]})

    y_pred_new = model.predict(x_new)
    result = output_encoder.inverse_transform(y_pred_new)
    return result[0]

# Streamlit UI
st.title("Heart Disease Prediction")

age = st.slider("Age", 0, 100, 50)
sex = st.selectbox("Sex", [0, 1])
cp = st.slider("Chest Pain Type", 0, 3, 1)
trtbps = st.slider("Resting Blood Pressure", 0, 200, 120)
chol = st.slider("Cholesterol", 0, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.slider("Resting Electrocardiographic Results", 0, 2, 0)
thalachh = st.slider("Maximum Heart Rate Achieved", 0, 250, 150)
exng = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.2, 3.0)
slp = st.slider("Slope of the Peak Exercise ST Segment", 0, 2, 1)
caa = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 1)
thall = st.slider("Thalassemia", 0, 3, 2)

if st.button("Predict"):
    result = predict_species(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)
    st.write(f"Predicted Specie: {result}")
