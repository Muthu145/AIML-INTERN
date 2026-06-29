# 🚗 Car Price Predictor & Family Recommender (ML)

Compares **Petrol**, **Diesel**, and **EV** cars on price + performance, and
recommends the best car for a family based on their **annual salary**.

---

## 📁 Files

| File                  | Purpose                                                              |
|-----------------------|-----------------------------------------------------------------------|
| `generate_dataset.py` | Builds a realistic synthetic dataset of 1,800 cars (600 each fuel type) |
| `train_model.py`      | Trains a Random Forest ML model to predict car price from specs       |
| `recommend_car.py`    | **Main script** — asks for salary, predicts/filters cars, recommends best |
| `car_dataset.csv`     | The generated dataset (created after step 1)                          |
| `car_price_model.pkl` | The trained, saved ML model (created after step 2)                    |

---

## 🛠️ Step 1 — Install requirements

```bash
pip install pandas numpy scikit-learn joblib
```

## 🛠️ Step 2 — Generate the dataset

```bash
python3 generate_dataset.py
```
This creates `car_dataset.csv` with realistic Petrol/Diesel/EV specs:
brand, year, engine size or battery size, power (bhp), mileage/range,
seats, yearly maintenance cost, and price (in ₹ lakhs).

> **Want REAL data instead?** Download a dataset like Kaggle's
> "Vehicle Dataset" or "Car Price Prediction", rename its columns to match
> (`brand, fuel_type, year, engine_cc, power_bhp, mileage_kmpl, battery_kwh,
> range_km, seats, maintenance_cost_per_year, price_lakhs`), save it as
> `car_dataset.csv`, and skip this step.

## 🛠️ Step 3 — Train the ML model

```bash
python3 train_model.py
```
This trains a `RandomForestRegressor` that learns the relationship between
a car's specs and its price. It prints accuracy (MAE and R²) and saves
the trained model to `car_price_model.pkl`.

## 🛠️ Step 4 — Get a recommendation

```bash
python3 recommend_car.py
```
You'll be asked for:
- **Annual family salary** (in ₹ lakhs, e.g. `12` for ₹12,00,000/year)
- **Family size** (number of members — used to ensure enough seats)

It will then:
1. Calculate an affordable car budget (default: **45% of annual salary** —
   a common personal-finance rule of thumb. Edit `AFFORDABILITY_RATIO` in
   `recommend_car.py` to change this).
2. Filter cars within that budget.
3. Score every car on a **Performance Score (0-100)** that blends power,
   mileage/range, and running cost.
4. Show you the **best Petrol pick**, **best Diesel pick**, and **best EV
   pick** — then declares one **overall winner** for your family.

---

## 🧠 How the recommendation logic works

```
Budget          = Annual Salary × 45%
Performance     = 0.5×Power + Efficiency_term − Maintenance_penalty
Value Score     = Performance ÷ Price  (normalized)
Final Score     = 0.6×Performance + 0.4×Value
```

- For **EVs**: efficiency uses **range (km)**.
- For **Petrol/Diesel**: efficiency uses **mileage (km/l)**.
- Cars seating fewer people than your family size are filtered out.
- The car with the highest **Final Score within budget** wins.

---

## ✏️ Customizing

- **Change affordability assumption**: edit `AFFORDABILITY_RATIO` (e.g. `0.30`
  for a more conservative 30% of salary, or `0.6` for aggressive).
- **Add real brands/models**: extend `generate_dataset.py`'s brand lists, or
  swap in a real CSV (see Step 2 note above).
- **Predict price for one specific car**: use the `predict_price()` function
  inside `recommend_car.py` — pass a dict like:
  ```python
  predict_price({
      "brand": "Hyundai", "fuel_type": "EV", "year": 2024,
      "engine_cc": None, "power_bhp": 140, "mileage_kmpl": None,
      "battery_kwh": 40, "range_km": 320, "seats": 5,
      "maintenance_cost_per_year": 6000
  })
  ```

---

## 📊 Example Output

```
Annual Family Salary : ₹30.00 lakhs/year
Recommended Max Car Budget (~45% of salary): ₹13.50 lakhs

BEST PICK PER FUEL TYPE:
  [Diesel] Skoda (2016) — ₹13.17 lakhs — Performance Score: 100.0
  [EV]     BYD (2019)   — ₹10.15 lakhs — Performance Score: 25.4
  [Petrol] Datsun (2019)— ₹5.31 lakhs  — Performance Score: 56.8

OVERALL BEST CHOICE: Diesel | Skoda (2016) — ₹13.17 lakhs
```
