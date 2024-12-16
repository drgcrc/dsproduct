import pandas as pd
import json
from app.preprocesses import load_data, preprocess_data, split_data
from app.model import train_model, save_model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = load_data("data/housing.csv")

X, y, preprocessor = preprocess_data(data)
X_train, X_test, y_train, y_test = split_data(X, y)


test_data = pd.DataFrame(X_test)
test_data["median_house_value"] = y_test
test_data.to_csv("app/test_data.csv", index=False)


model = train_model(X_train, y_train)


save_model(
    model,
    preprocessor,
    model_path="app/model.joblib",
    preprocessor_path="app/preprocessor.joblib",
)

print("Model and preprocessor saved successfully.")

# Evaluate model on the test set
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# Save evaluation results
performance_data = {
    "mse": mse,
    "rmse": rmse,
    "r2": r2,
    "actual": list(y_test),
    "predicted": list(predictions),
}

with open("app/performance.json", "w") as f:
    json.dump(performance_data, f)

print(f"Performance Metrics:\nMSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
