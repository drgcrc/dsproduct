import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("California Housing Price Predictor")

st.write("Enter the housing features to predict the price:")

# Input fields
longitude = st.number_input("Longitude", value=-122.2)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41)
total_rooms = st.number_input("Total Rooms", value=880)
total_bedrooms = st.number_input("Total Bedrooms", value=129)
population = st.number_input("Population", value=322)
households = st.number_input("Households", value=126)
median_income = st.number_input("Median Income", value=8.3252)
ocean_proximity = st.selectbox("Ocean Proximity", ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"])

# Make prediction
if st.button("Predict Price"):
    input_data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    # Log the input data
    st.write("Input Data:")
    st.write(input_data)

    response = requests.post("http://127.0.0.1:5000/predict", json={"input_data": input_data})

    # Log the response status code and content
    st.write("Response Status Code:")
    st.write(response.status_code)
    st.write("Response Content:")
    st.write(response.json())

    if response.status_code == 200:
        predicted_price = response.json().get("predicted_price", "N/A")
        st.success(f"Predicted House Price: ${predicted_price:.2f}")
    else:
        st.error(f"Error: {response.json().get('error', 'Unknown error')}")