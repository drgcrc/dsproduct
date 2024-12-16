import requests

def test_predict():
    url = "http://127.0.0.1:5000/predict"
    input_data = {
        "longitude": -122.2,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }

    response = requests.post(url, json={"input_data": input_data})

    print("Response Status Code:", response.status_code)
    print("Response Content:", response.json())

if __name__ == "__main__":
    test_predict()