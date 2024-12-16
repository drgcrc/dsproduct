from flask import Flask, request, jsonify
from app.model import load_model
import pandas as pd


app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = "app/model.joblib"
PREPROCESSOR_PATH = "app/preprocessor.joblib"
try:
    model, preprocessor = load_model(MODEL_PATH, PREPROCESSOR_PATH)
except FileNotFoundError:
    model, preprocessor = None, None
    print("Error: Model and preprocessor not found. Please train the model first.")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or preprocessor is None:
        return jsonify(
            {"error": "Model not available. Please train the model first."}
        ), 500
    try:
        # Parse input JSON
        input_data = request.get_json().get("input_data")
        if not input_data:
            return jsonify({"error": "Invalid input format"}), 400

        # Convert input to appropriate format
        column_names = [
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income",
            "ocean_proximity",
        ]
        input_df = pd.DataFrame([input_data], columns=column_names)

        # Print the shape of input_df before preprocessing
        print("Input DataFrame Shape (Before Preprocessing):", input_df.shape)
        print("Input DataFrame (Before Preprocessing):")
        print(input_df)

        # Preprocess input
        processed_input = preprocessor.transform(input_df)

        # Print the shape of processed_input after preprocessing
        print("Processed Input Shape (After Preprocessing):", processed_input.shape)
        print("Processed Input (After Preprocessing):")
        print(processed_input)

        # Predict using the model
        prediction = model.predict(processed_input)

        # Return prediction as JSON
        return jsonify({"predicted_price": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
