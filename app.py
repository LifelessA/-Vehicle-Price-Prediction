# app.py

import streamlit as st
import pandas as pd
import joblib

# --- Load Model and Data ---
try:
    pipeline = joblib.load('vehicle_price_pipeline.pkl')
    model_columns_info = joblib.load('model_columns.pkl')
    model_columns = model_columns_info['columns']
    categorical_options = model_columns_info['categorical_options']
    # Load the original dataset to get options for numerical dropdowns
    df_original = pd.read_csv('dataset.csv')
    
except FileNotFoundError:
    st.error("Model files not found. Please run the Jupyter Notebook first to generate them.")
    st.stop()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Vehicle Price Predictor", layout="wide")

# App title
st.title("🚗 Vehicle Price Prediction")
st.markdown("Enter the vehicle's specifications to get an estimated price.")

# --- User Input Section ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Manufacturer Details")
    make = st.selectbox("Make", sorted(categorical_options['make']))
    
    # Filter model options based on the selected make
    model_options = df_original[df_original['make'] == make]['model'].unique().tolist()
    model = st.selectbox("Model", sorted(model_options))
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2024, step=1)
    
with col2:
    st.header("Performance Specs")
    cylinder_options = sorted(df_original['cylinders'].dropna().unique().astype(int))
    cylinders = st.selectbox("Cylinders", cylinder_options)
    fuel = st.selectbox("Fuel Type", sorted(categorical_options['fuel']))
    mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=10000, step=100)

with col3:
    st.header("Body & Drivetrain")
    transmission = st.selectbox("Transmission", sorted(categorical_options['transmission']))
    body = st.selectbox("Body Style", sorted(categorical_options['body']))
    drivetrain = st.selectbox("Drivetrain", sorted(categorical_options['drivetrain']))
    # --- THIS IS THE CORRECTED LINE ---
    door_options = sorted(df_original['doors'].dropna().unique().astype(int))
    doors = st.selectbox("Doors", door_options)

# --- Prediction Logic ---
if st.button("Predict Price", use_container_width=True):
    # Create a DataFrame from the user inputs
    input_data = {
        'make': [make],
        'model': [model],
        'year': [year],
        'cylinders': [cylinders],
        'fuel': [fuel],
        'mileage': [mileage],
        'transmission': [transmission],
        'body': [body],
        'doors': [doors],
        'drivetrain': [drivetrain]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Ensure columns are in the correct order
    input_df = input_df[model_columns]

    # Make prediction
    try:
        prediction = pipeline.predict(input_df)
        predicted_price = prediction[0]
        
        st.success(f"**Predicted Price: ₹{predicted_price*83.50:,.2f}**")
        st.balloons()
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
