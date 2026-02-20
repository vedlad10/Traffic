from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from datetime import datetime
import math

app = FastAPI()

# ✅ ENABLE CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("routing_risk_model.pkl")
encoders = joblib.load("routing_label_encoders.pkl")


class RouteInput(BaseModel):
    coordinates: list


# ---------------- GEOMETRY RISK ---------------- #

def calculate_geometry_risk(coordinates):

    sharp_turns = 0
    total_turns = 0

    for i in range(1, len(coordinates) - 1):

        lon1, lat1 = coordinates[i - 1]
        lon2, lat2 = coordinates[i]
        lon3, lat3 = coordinates[i + 1]

        v1 = (lon2 - lon1, lat2 - lat1)
        v2 = (lon3 - lon2, lat3 - lat2)

        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if mag1 == 0 or mag2 == 0:
            continue

        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        angle = math.degrees(math.acos(cos_angle))

        total_turns += 1

        if angle < 120:
            sharp_turns += 1

    if total_turns == 0:
        return 0

    sharp_density = sharp_turns / total_turns

    return min(sharp_density * 0.3, 0.3)


# ---------------- SAFE ROUTE ENDPOINT ---------------- #

@app.post("/safe-route")
def safe_route(data: RouteInput):

    if not data.coordinates or len(data.coordinates) < 3:
        return {"error": "Not enough route coordinates"}

    # 1️⃣ Midpoint Weather Sampling
    mid_index = len(data.coordinates) // 2
    mid_lat = data.coordinates[mid_index][1]
    mid_lon = data.coordinates[mid_index][0]

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={mid_lat}&longitude={mid_lon}"
        f"&current=precipitation,cloudcover,windspeed_10m,visibility"
        f"&timezone=auto"
    )

    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()

    current_data = weather_data.get("current", {})

    precipitation = current_data.get("precipitation", 0)
    cloudcover = current_data.get("cloudcover", 0)
    windspeed = current_data.get("windspeed_10m", 0)
    visibility = current_data.get("visibility", 20000)

    # 2️⃣ Rain Risk (Non-linear Bands)
    if precipitation == 0:
        rain_risk = 0
    elif precipitation < 2:
        rain_risk = 0.05
    elif precipitation < 7:
        rain_risk = 0.12
    elif precipitation < 15:
        rain_risk = 0.20
    else:
        rain_risk = 0.30

    # 3️⃣ Wind Risk
    if windspeed < 10:
        wind_risk = 0
    elif windspeed < 25:
        wind_risk = 0.05
    elif windspeed < 40:
        wind_risk = 0.12
    else:
        wind_risk = 0.20

    # 4️⃣ Cloud Risk (Minor Contributor)
    cloud_risk = (cloudcover / 100) * 0.1

    # 5️⃣ Visibility Risk (Threshold-Based)
    if visibility > 25000:
        visibility_risk = 0
    elif visibility > 15000:
        visibility_risk = 0.03
    elif visibility > 10000:
        visibility_risk = 0.08
    elif visibility > 5000:
        visibility_risk = 0.15
    elif visibility > 2000:
        visibility_risk = 0.22
    else:
        visibility_risk = 0.30

    # 6️⃣ Night Detection (Simple + Stable)
    hour = datetime.now().hour
    night_risk = 0.2 if (hour >= 19 or hour <= 5) else 0.05

    # 7️⃣ Geometry Risk
    geometry_risk = calculate_geometry_risk(data.coordinates)

    # 8️⃣ Historical ML Risk
    total_base = 0

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
            "Hour": hour,
            "Road_Surface_Conditions": 0
        }

        df = pd.DataFrame([features])
        base_risk = model.predict(df)[0]
        total_base += base_risk

    avg_base = total_base / len(data.coordinates)

    # 9️⃣ Final Risk Formula (Balanced Weights)
    final_risk = (
        0.30 * avg_base +
        0.20 * rain_risk +
        0.15 * visibility_risk +
        0.10 * wind_risk +
        0.15 * geometry_risk +
        0.10 * night_risk +
        0.05 * cloud_risk
    )

    weather_risk = rain_risk + wind_risk + cloud_risk + visibility_risk

    return {
        "average_risk": float(final_risk),
        "historical_risk": float(avg_base),
        "weather_risk": float(weather_risk),  # ← ADD THIS BACK
        "rain_component": float(rain_risk),
        "wind_component": float(wind_risk),
        "cloud_component": float(cloud_risk),
        "visibility_risk": float(visibility_risk),
        "visibility_meters": float(visibility),
        "night_risk": float(night_risk),
        "geometry_risk": float(geometry_risk),
        "formula": "0.30*Historical + 0.20*Rain + 0.15*Visibility + 0.10*Wind + 0.15*Geometry + 0.10*Night + 0.05*Cloud"
    }