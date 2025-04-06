import streamlit as st
import pandas as pd
import joblib
from explainer import explain_prediction

model = joblib.load("../models/model.pkl")

st.title("AI-Powered Financial Fraud Detector")

uploaded_file = st.file_uploader("Upload a transaction CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Transaction Preview:")
    st.dataframe(data)

    if st.button("Predict Fraud"):
        for idx, row in data.iterrows():
            prediction = model.predict([row])[0]
            explanation = explain_prediction(row, prediction)

            st.subheader(f"Transaction {idx + 1}")
            st.write(f"**Prediction**: {'Fraudulent' if prediction == 1 else 'Legitimate'}")
            st.write("**LLM Explanation:**")
            st.info(explanation)
