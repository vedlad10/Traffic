from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# ✅ ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (safe for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("routing_risk_model.pkl")
encoders = joblib.load("routing_label_encoders.pkl")

class RouteInput(BaseModel):
    coordinates: list
    rain_intensity: float
    visibility_score: float

@app.post("/safe-route")
def safe_route(data: RouteInput):

    total_risk = 0

    for coord in data.coordinates:

        lat = coord[1]
        lon = coord[0]

        features = {
            "lat_grid": round(lat, 3),
            "lon_grid": round(lon, 3),
            "Number_of_Casualties": 1,
            "Number_of_Vehicles": 1,
            "Age_of_Driver": 30,
            "Engine_Capacity_(CC)": 1500,
            "Age_of_Vehicle": 5,
            "Hour": 20,
            "Road_Surface_Conditions": 0
        }

        df = pd.DataFrame([features])
        base_risk = model.predict(df)[0]

        final_risk = (
            base_risk
            + 0.2 * data.rain_intensity
            + 0.15 * (1 - data.visibility_score)
        )

        total_risk += final_risk

    avg_risk = total_risk / len(data.coordinates)

    return {"average_risk": float(avg_risk)}