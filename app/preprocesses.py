import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(filepath):
    data = pd.read_csv(filepath)
    return data


def preprocess_data(data):
    # Handle missing values in `total_bedrooms` column
    data["total_bedrooms"].fillna(data["total_bedrooms"].median(), inplace=True)

    # One-hot encode the `ocean_proximity` column
    categorical_features = ["ocean_proximity"]
    numerical_features = data.drop(
        columns=["ocean_proximity", "median_house_value"]
    ).columns

    print("Numerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(), categorical_features),
        ]
    )

    X = data.drop(columns=["median_house_value"])
    y = data["median_house_value"]

    print("Input DataFrame before preprocessing:")
    print(X.head())
    print("Input DataFrame Shape (before preprocessing):", X.shape)

    X = preprocessor.fit_transform(X)

    print("Transformed Data Shape (after preprocessing):", X.shape)
    print("Transformed Data (first 5 rows):")
    print(X[:5])  # Show first 5 rows of transformed data

    return X, y, preprocessor


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
