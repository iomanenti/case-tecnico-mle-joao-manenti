import unittest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running"}

class TestModelLoad(unittest.TestCase):
    def test_load_model(self):
        
        with open("models/linear_regression_model_flight_delay_prediction.pkl", "rb") as model_file:
            response = client.post("/model/load/", files={"file": model_file})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["message"], "Model loaded successfully")

class TestModelPredict(unittest.TestCase):
    def test_predict(self):
        # Payload data for prediction
        payload = {
            "year": 2024,
            "month": 10,
            "day": 7,
            "dep_time": 1422,
            "sched_dep_time": 1543,
            "dep_delay": 15.0,
            "arr_delay": 5.0,
            "carrier": "AA",
            "flight": 123,
            "origin": "JFK",
            "dest": "LAX",
            "dep_delay_label": 1,
            "arr_delay_label": 1,
            "wind_speed_origin":  4.34,
            "wind_speed_dest": 1.41,
            "route": "JFK-LAX",
            "route_index" : 2,
            "distance": 1400
        }

        response = client.post("/model/predict/", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)  # Check if response contains predicted delay

class TestModelHistory(unittest.TestCase):
    def test_history(self):
        response = client.post("/model/history/")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)  # Check if history is returned as a list


