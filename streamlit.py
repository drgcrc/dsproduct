import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json

st.title("California Housing Price Predictor")

# Navigation
section = st.radio("Choose Section", ["Predict", "Analyze Performance"])

if section == "Predict":
    st.header("Predict Housing Prices")

    # Input fields
    longitude = st.number_input("Longitude", value=-122.2)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Housing Median Age", value=41)
    total_rooms = st.number_input("Total Rooms", value=880)
    total_bedrooms = st.number_input("Total Bedrooms", value=129)
    population = st.number_input("Population", value=322)
    households = st.number_input("Households", value=126)
    median_income = st.number_input("Median Income", value=8.3252)
    ocean_proximity = st.selectbox(
        "Ocean Proximity", ["NEAR BAY", "INLAND", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]
    )

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
            "ocean_proximity": ocean_proximity,
        }

        # Log the input data
        st.write("Input Data:")
        st.write(input_data)

        response = requests.post(
            "http://127.0.0.1:5000/predict", json={"input_data": input_data}
        )

        # Log the response status code and content
        st.write("Response Status Code:")
        st.write(response.status_code)
        st.write("Response Content:")
        st.write(response.json())

        if response.status_code == 200:
            predicted_price = response.json().get("predicted_price", "N/A")
            st.success(f"Predicted House Price: ${predicted_price:,.2f}")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")

elif section == "Analyze Performance":
    st.header("Analyze Model Performance")

    # Load performance data
    with open("app/performance.json", "r") as f:
        performance_data = json.load(f)

    # Display metrics
    st.subheader("Model Performance Metrics")
    st.write(f"**Mean Squared Error (MSE):** {performance_data['mse']:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {performance_data['rmse']:.2f}")
    st.write(f"**R² Score:** {performance_data['r2']:.2f}")

    # Visualization: Predicted vs Actual
    st.subheader("Predicted vs Actual Values")
    actual = performance_data["actual"]
    predicted = performance_data["predicted"]

    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=0.6)
    ax.plot(
        [min(actual), max(actual)],
        [min(actual), max(actual)],
        color="red",
        linestyle="--",
    )
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Predicted vs Actual Prices")
    st.pyplot(fig)

    # Feature importance (optional for models like Random Forest)
    st.subheader("Feature Importance")
    try:
        model_path = "app/model.joblib"
        from joblib import load

        model = load(model_path)
        feature_importance = model.named_steps["regressor"].feature_importances_

        numeric_features = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
        ]
        features = numeric_features + list(
            model.named_steps["preprocessor"]
            .transformers_[1][1]
            .get_feature_names_out()
        )
        feature_df = pd.DataFrame(
            {"Feature": features, "Importance": feature_importance}
        ).sort_values(by="Importance", ascending=False)

        st.bar_chart(feature_df.set_index("Feature"))
    except AttributeError:
        st.write("Feature importance not supported by the current model.")
