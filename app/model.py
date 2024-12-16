from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2, predictions


def save_model(model, preprocessor, model_path="app/model.joblib", preprocessor_path="app/preprocessor.joblib"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)


def load_model(model_path="app/model.joblib", preprocessor_path="app/preprocessor.joblib"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor