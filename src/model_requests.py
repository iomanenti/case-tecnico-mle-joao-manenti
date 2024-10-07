import requests

def load_model():

    url = "http://localhost:8000/model/load/"

    file_path = "C:/models/linear_regression_model_flight_delay_prediction.pkl"

    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    print(response.json())

def predict_model():

    # Define the URL of your FastAPI endpoint
    url = "http://127.0.0.1:8000/model/predict/"

    # Define the payload for the request (match the structure of FlightPayload)
    payload = {
        "year": 2023,
        "month": 10,
        "day": 7,
        "dep_time": 1400,
        "sched_dep_time": 1345,
        "dep_delay": 15.0,
        "arr_delay": 5.0,
        "carrier": "AA",
        "flight": 123,
        "origin": "JFK",
        "dest": "LAX",
        "dep_delay_label": 1,
        "arr_delay_label": 1,
        "wind_speed_origin":  5.34,
        "wind_speed_dest": 3.41,
        "route": "JFK-LAX",
        "route_index" : 2,
        "distance": 1400
    }

    # Send a POST request to the predict endpoint
    response = requests.post(url, json=payload)

    # Print the prediction result
    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)

# Running the API
if __name__ == "__main__":
    predict_model()