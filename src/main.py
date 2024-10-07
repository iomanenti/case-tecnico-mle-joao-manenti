import pickle
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
import json
from fastapi.responses import JSONResponse

app = FastAPI()

# Global variable to store the model data (coefficients and intercept)
model_data = None

# Global variable to store the path of the history file
history_file_path = "prediction_history.json"

class FlightPayload(BaseModel):
    year: int
    month: int
    day: int
    dep_time: int
    sched_dep_time: int
    dep_delay: float
    arr_delay: float
    carrier: str
    flight: int
    origin: str
    dest: str
    dep_delay_label: int
    arr_delay_label: int
    wind_speed_origin:  float
    wind_speed_dest: float
    route: str
    route_index: int
    distance: int

# Function to manually apply the linear regression formula: y = X * coefficients + intercept
def manual_predict(features, coefficients, intercept):
    return np.dot(features, coefficients) + intercept

@app.post("/model/load/")
async def load_model(file: UploadFile = File(...)):
    """
    Endpoint to load the model from a pickle file.
    """
    global model_data
    model_data = pickle.loads(await file.read())
        
    return JSONResponse(content={"message": "Model loaded successfully"}, status_code=200)

@app.post("/model/predict/")
def predict(payload: FlightPayload):
    """
    Endpoint to make predictions using the loaded model data (coefficients and intercept).
    """
    if model_data is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    # Prepare features as a list of numeric values based on the payload
    features = np.array([[
        payload.dep_delay_label, 
        payload.wind_speed_origin, 
        payload.wind_speed_dest, 
        payload.distance,
        payload.route_index
    ]])

    # Ensure the model has coefficients and intercept
    coefficients = model_data["coefficients"]
    intercept = model_data["intercept"]
    
    # Use the manual prediction function to compute the prediction
    prediction = manual_predict(features, coefficients, intercept)

    # Convert prediction (which is a NumPy array) to a Python list
    prediction_value = prediction.tolist()

    # Save the payload and prediction to the history file
    with open(history_file_path, "a") as history_file:
        history_entry = {
            "payload": payload.dict(),
            "prediction": prediction_value
        }
        history_file.write(json.dumps(history_entry) + "\n")

    return JSONResponse(content={"prediction": prediction_value}, status_code=200)

@app.post("/model/history/")
def get_history():
    """
    Endpoint to retrieve the prediction history.
    """
    try:
        with open(history_file_path, "r") as history_file:
            history = [json.loads(line) for line in history_file]
        return JSONResponse(content={"history": history}, status_code=200)
    except FileNotFoundError:
        return JSONResponse(content={"history": []}, status_code=200)

@app.get("/health/")
def health_check():
    """
    Health check endpoint.
    """
    return JSONResponse(content={"status": "API is running"}, status_code=200)

# Running the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)