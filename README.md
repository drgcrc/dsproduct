# dsproduct

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd dsproduct
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

### Train the Model

1. Run the [main.py](http://_vscodecontentref_/0) script to train the model and save the preprocessor:
    ```sh
    python main.py
    ```

### Start the API

1. Run the `api.py` script to start the Flask API:
    ```sh
    python app/api.py
    ```

2. The API will be available at `http://127.0.0.1:5000`.

### Make Predictions

1. Use a tool like `curl` or Postman to make POST requests to the `/predict` endpoint with the input data in JSON format:
    ```sh
    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"input_data": {"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41, "total_rooms": 880, "total_bedrooms": 129, "population": 322, "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}}'
    ```

### Run the Streamlit App

1. Run the [streamlit.py](http://_vscodecontentref_/1) script to start the Streamlit app:
    ```sh
    streamlit run streamlit.py
    ```

2. The Streamlit app will be available at `http://localhost:8501`.

## Running Tests

1. Run the [test_api.py](http://_vscodecontentref_/2) script to execute the tests:
    ```sh
    python test_api.py
    ```