"""
generate_dataset.py
--------------------
Generates a synthetic but realistic car dataset covering Petrol, Diesel,
and EV cars with price + performance specs.

If you have a REAL dataset (e.g. from Kaggle "Car Price Prediction" or
"Vehicle Dataset"), just replace this step — load your CSV instead and
make sure it has similar column names (see bottom of this file).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_TYPE = 600  # number of synthetic cars per fuel type

brands_budget = ["Maruti Suzuki", "Tata", "Hyundai", "Renault", "Datsun"]
brands_mid = ["Honda", "Toyota", "Hyundai", "Kia", "Skoda", "Volkswagen"]
brands_premium = ["BMW", "Audi", "Mercedes-Benz", "Volvo", "Jaguar"]
ev_brands = ["Tata", "MG", "Hyundai", "BYD", "Mahindra", "Tesla"]

rows = []

def rand_choice_weighted(options, weights):
    return np.random.choice(options, p=np.array(weights) / sum(weights))

# ---------------- PETROL ----------------
for _ in range(N_PER_TYPE):
    segment = rand_choice_weighted(["budget", "mid", "premium"], [0.5, 0.35, 0.15])
    if segment == "budget":
        brand = np.random.choice(brands_budget)
        engine_cc = np.random.randint(800, 1300)
        power_bhp = np.random.randint(55, 95)
        base_price = np.random.uniform(4.5, 9.0)       # lakhs INR
    elif segment == "mid":
        brand = np.random.choice(brands_mid)
        engine_cc = np.random.randint(1300, 2000)
        power_bhp = np.random.randint(95, 160)
        base_price = np.random.uniform(9.0, 18.0)
    else:
        brand = np.random.choice(brands_premium)
        engine_cc = np.random.randint(1800, 3000)
        power_bhp = np.random.randint(160, 280)
        base_price = np.random.uniform(35.0, 80.0)

    year = np.random.randint(2015, 2025)
    mileage_kmpl = round(np.random.uniform(12, 23) - (power_bhp / 100), 1)
    mileage_kmpl = max(8, mileage_kmpl)
    seats = np.random.choice([5, 5, 5, 7])
    maintenance_cost_per_year = round(np.random.uniform(8000, 25000), -2)
    price_lakhs = round(base_price * (1 + (year - 2015) * 0.015) * (1 + power_bhp / 1000), 2)

    rows.append([brand, "Petrol", year, engine_cc, power_bhp, mileage_kmpl,
                 None, None, seats, maintenance_cost_per_year, price_lakhs])

# ---------------- DIESEL ----------------
for _ in range(N_PER_TYPE):
    segment = rand_choice_weighted(["budget", "mid", "premium"], [0.35, 0.45, 0.20])
    if segment == "budget":
        brand = np.random.choice(brands_budget)
        engine_cc = np.random.randint(1200, 1500)
        power_bhp = np.random.randint(70, 100)
        base_price = np.random.uniform(6.5, 11.0)
    elif segment == "mid":
        brand = np.random.choice(brands_mid)
        engine_cc = np.random.randint(1500, 2200)
        power_bhp = np.random.randint(100, 180)
        base_price = np.random.uniform(11.0, 22.0)
    else:
        brand = np.random.choice(brands_premium)
        engine_cc = np.random.randint(2000, 3000)
        power_bhp = np.random.randint(180, 300)
        base_price = np.random.uniform(40.0, 90.0)

    year = np.random.randint(2015, 2025)
    mileage_kmpl = round(np.random.uniform(16, 28) - (power_bhp / 120), 1)
    mileage_kmpl = max(10, mileage_kmpl)
    seats = np.random.choice([5, 5, 7, 7])
    maintenance_cost_per_year = round(np.random.uniform(10000, 30000), -2)
    price_lakhs = round(base_price * (1 + (year - 2015) * 0.015) * (1 + power_bhp / 1000), 2)

    rows.append([brand, "Diesel", year, engine_cc, power_bhp, mileage_kmpl,
                 None, None, seats, maintenance_cost_per_year, price_lakhs])

# ---------------- EV ----------------
for _ in range(N_PER_TYPE):
    segment = rand_choice_weighted(["budget", "mid", "premium"], [0.30, 0.45, 0.25])
    brand = np.random.choice(ev_brands)
    if segment == "budget":
        battery_kwh = np.random.uniform(20, 35)
        power_bhp = np.random.randint(60, 100)
        base_price = np.random.uniform(8.0, 14.0)
    elif segment == "mid":
        battery_kwh = np.random.uniform(35, 55)
        power_bhp = np.random.randint(100, 180)
        base_price = np.random.uniform(14.0, 28.0)
    else:
        battery_kwh = np.random.uniform(55, 100)
        power_bhp = np.random.randint(180, 500)
        base_price = np.random.uniform(45.0, 120.0)

    year = np.random.randint(2018, 2025)
    range_km = round(battery_kwh * np.random.uniform(5.5, 7.5), 0)  # rough Wh/km efficiency
    seats = np.random.choice([5, 5, 7])
    # EVs cost far less per year to "fuel" + service
    maintenance_cost_per_year = round(np.random.uniform(4000, 12000), -2)
    price_lakhs = round(base_price * (1 + (year - 2018) * 0.02) * (1 + power_bhp / 1500), 2)

    rows.append([brand, "EV", year, None, power_bhp, None,
                 round(battery_kwh, 1), range_km, seats, maintenance_cost_per_year, price_lakhs])

df = pd.DataFrame(rows, columns=[
    "brand", "fuel_type", "year", "engine_cc", "power_bhp", "mileage_kmpl",
    "battery_kwh", "range_km", "seats", "maintenance_cost_per_year", "price_lakhs"
])

df.to_csv("/home/claude/car_dataset.csv", index=False)
print("Dataset created:", df.shape)
print(df.groupby("fuel_type")["price_lakhs"].describe())
