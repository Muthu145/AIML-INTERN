"""
train_model.py
---------------
Trains a machine learning model to PREDICT car price (in lakhs INR)
from specs, for Petrol, Diesel, and EV cars.

Model: RandomForestRegressor (handles non-linear relationships + mixed
categorical/numeric features well, and works great on small/medium data).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load data ---------------------------------------------------------
df = pd.read_csv("/home/claude/car_dataset.csv")

TARGET = "price_lakhs"

categorical_features = ["brand", "fuel_type"]
numeric_features = ["year", "engine_cc", "power_bhp", "mileage_kmpl",
                     "battery_kwh", "range_km", "seats", "maintenance_cost_per_year"]

X = df[categorical_features + numeric_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Preprocessing -------------------------------------------------------
# Numeric: fill missing (EVs have no engine_cc/mileage, petrol/diesel have no battery/range)
#          then scale.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 3. Full model pipeline ----------------------------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    ))
])

# 4. Train ---------------------------------------------------------------
model.fit(X_train, y_train)

# 5. Evaluate --------------------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"Model Performance on Test Set:")
print(f"  MAE  (avg error) : ₹{mae:.2f} lakhs")
print(f"  R²   (fit score) : {r2:.3f}  (closer to 1.0 = better)")

# 6. Save model + dataset for reuse --------------------------------------
joblib.dump(model, "/home/claude/car_price_model.pkl")
print("\nModel saved to car_price_model.pkl")
