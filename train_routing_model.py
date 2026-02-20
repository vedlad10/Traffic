import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load data
df = pd.read_excel("VehiclesBig.xlsx")

# Remove missing coordinates
df = df.dropna(subset=["latitude", "longitude"])

# Create spatial grid (approx 500m resolution)
grid_size = 0.005
df["lat_grid"] = (df["latitude"] / grid_size).round() * grid_size
df["lon_grid"] = (df["longitude"] / grid_size).round() * grid_size

# Normalize severity to 0-1
df["Normalized_Severity"] = df["Accident_Severity"] / df["Accident_Severity"].max()

# Convert Time to hour
df["Hour"] = pd.to_datetime(df["Time"], errors="coerce").dt.hour

# Encode categorical features
categorical_cols = [
    "Sex_of_Driver",
    "Sex_of_Casualty",
    "Road_Surface_Conditions",
    "Day_of_Week"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Aggregate per grid
grouped = df.groupby(["lat_grid", "lon_grid"]).agg({
    "Normalized_Severity": "mean",
    "Number_of_Casualties": "sum",
    "Number_of_Vehicles": "mean",
    "Age_of_Driver": "mean",
    "Engine_Capacity_(CC)": "mean",
    "Age_of_Vehicle": "mean",
    "Hour": "mean",
    "Road_Surface_Conditions": "mean"
}).reset_index()

# Define features & target
X = grouped.drop(["Normalized_Severity"], axis=1)
y = grouped["Normalized_Severity"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "routing_risk_model.pkl")
joblib.dump(encoders, "routing_label_encoders.pkl")

print("Routing model trained and saved.")