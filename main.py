import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from app.preprocesses import load_data, preprocess_data, split_data
from app.model import train_model, evaluate_model, save_model

# Step 1: Load the dataset
data = load_data("data/housing.csv")

# Step 2: Preprocess the data
X, y, preprocessor = preprocess_data(data)

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Step 4: Save the test set for evaluation later
test_data = pd.DataFrame(X_test)
test_data["median_house_value"] = y_test
test_data.to_csv("app/test_data.csv", index=False)

# Step 5: Train the model
model = train_model(X_train, y_train)

# Step 6: Save the trained model and preprocessor
save_model(model, preprocessor, model_path="app/model.joblib", preprocessor_path="app/preprocessor.joblib")

print("Model and preprocessor saved successfully.")