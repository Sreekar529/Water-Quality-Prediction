import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and column structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Inject custom CSS for water-themed design
st.markdown("""
    <style>
        body {
            background-color: #e0f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #0277bd;
            text-align: center;
        }
        .stButton > button {
            background-color: #0288d1;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #01579b;
        }
        .stNumberInput, .stTextInput {
            background-color: #e1f5fe;
            padding: 8px;
            border-radius: 6px;
        }
        .css-1v0mbdj, .css-1x8cf1d {
            background-color: #e0f7fa !important;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("💧 Water Pollutants Predictor")

st.write("### Enter the details to predict pollutant levels:")

year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

if st.button('Predict'):
    if not station_id:
        st.warning('⚠️ Please enter the station ID')
    else:
        # Prepare input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"🌊 Predicted pollutant levels for Station '{station_id}' in {year_input}:")
        for p, val in zip(pollutants, predicted_pollutants):
            st.markdown(f"- **{p}**: {val:.2f}")
