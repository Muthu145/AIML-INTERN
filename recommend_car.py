"""
recommend_car.py
------------------
Main user-facing script.

What it does:
1. Loads the trained ML price-prediction model + dataset.
2. Asks for the family's ANNUAL SALARY (income).
3. Works out a sensible car budget from that salary using the
   standard personal-finance rule of thumb:
        Affordable car price  =  40% to 50% of ANNUAL salary
   (You can change AFFORDABILITY_RATIO below if you want to be
   more conservative or more aggressive.)
4. Filters cars within budget, scores each one on PERFORMANCE
   (power, efficiency/range, running cost) and VALUE (performance
   per rupee).
5. Picks the single best Petrol, best Diesel, and best EV pick,
   then declares an overall WINNER for the family.

Run:  python3 recommend_car.py
"""

import pandas as pd
import numpy as np
import joblib

# ----------------------- CONFIG --------------------------------------
AFFORDABILITY_RATIO = 0.45   # max car price = 45% of annual salary (lakhs)
LAKH = 100_000                # 1 lakh = 100,000 INR

# ----------------------- LOAD ASSETS ----------------------------------
df = pd.read_csv("/home/claude/car_dataset.csv")
model = joblib.load("/home/claude/car_price_model.pkl")


def predict_price(row: dict) -> float:
    """Use the trained ML model to predict price (lakhs) for a single car spec."""
    input_df = pd.DataFrame([row])
    return float(model.predict(input_df)[0])


def performance_score(row) -> float:
    """
    A 0-100 performance score blending power, efficiency, and running cost.
    Different formula per fuel type since EV/petrol/diesel specs differ.
    """
    power = row["power_bhp"]
    maint = row["maintenance_cost_per_year"]

    if row["fuel_type"] == "EV":
        efficiency_term = (row["range_km"] / 10)         # range matters a lot for EVs
    else:
        efficiency_term = (row["mileage_kmpl"] * 4)       # mileage matters for ICE cars

    running_cost_penalty = maint / 1000                   # higher maintenance = worse

    raw_score = (power * 0.5) + efficiency_term - running_cost_penalty
    return raw_score


def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-9) * 100


def get_recommendations(annual_salary_lakhs: float, family_size: int = 4):
    budget = annual_salary_lakhs * AFFORDABILITY_RATIO

    candidates = df[df["price_lakhs"] <= budget].copy()
    candidates = candidates[candidates["seats"] >= min(family_size, 5)]  # basic seating need

    if candidates.empty:
        return None, budget, None

    candidates["perf_raw"] = candidates.apply(performance_score, axis=1)
    candidates["perf_score"] = normalize(candidates["perf_raw"])
    candidates["value_score"] = normalize(candidates["perf_score"] / (candidates["price_lakhs"] + 1))
    # Final score: blend of performance + value-for-money
    candidates["final_score"] = candidates["perf_score"] * 0.6 + candidates["value_score"] * 0.4

    best_per_fuel = (
        candidates.sort_values("final_score", ascending=False)
        .groupby("fuel_type")
        .first()
        .reset_index()
    )

    overall_best = candidates.sort_values("final_score", ascending=False).iloc[0]

    return best_per_fuel, budget, overall_best


def print_report(annual_salary_lakhs, family_size=4):
    print("=" * 70)
    print(f" FAMILY CAR RECOMMENDATION REPORT")
    print("=" * 70)
    print(f"Annual Family Salary : ₹{annual_salary_lakhs:.2f} lakhs/year")
    print(f"Family Size          : {family_size}")

    best_per_fuel, budget, overall_best = get_recommendations(annual_salary_lakhs, family_size)

    print(f"Recommended Max Car Budget (~{int(AFFORDABILITY_RATIO*100)}% of salary): "
          f"₹{budget:.2f} lakhs")
    print("-" * 70)

    if best_per_fuel is None:
        print("No cars found within this budget for the given family size.")
        print("Tip: increase savings, consider used cars, or a smaller segment.")
        return

    print("\nBEST PICK PER FUEL TYPE:\n")
    for _, row in best_per_fuel.iterrows():
        print(f"  [{row['fuel_type']}] {row['brand']}  |  Year: {int(row['year'])}")
        print(f"     Price          : ₹{row['price_lakhs']:.2f} lakhs")
        print(f"     Power          : {row['power_bhp']} bhp")
        if row["fuel_type"] == "EV":
            print(f"     Battery/Range  : {row['battery_kwh']} kWh / {row['range_km']} km")
        else:
            print(f"     Engine/Mileage : {row['engine_cc']} cc / {row['mileage_kmpl']} km-pl")
        print(f"     Annual upkeep  : ₹{row['maintenance_cost_per_year']:.0f}")
        print(f"     Performance Score (0-100): {row['perf_score']:.1f}")
        print()

    print("-" * 70)
    print(f"OVERALL BEST CHOICE FOR YOUR FAMILY:")
    print(f"   >>> {overall_best['fuel_type']} | {overall_best['brand']} "
          f"({int(overall_best['year'])}) — ₹{overall_best['price_lakhs']:.2f} lakhs <<<")
    print(f"   Why: Best balance of performance, running cost, and affordability")
    print(f"   for a ₹{annual_salary_lakhs:.1f} lakh/year household.")
    print("=" * 70)


# ----------------------- INTERACTIVE MODE -----------------------------
if __name__ == "__main__":
    try:
        salary_input = input("Enter family ANNUAL salary (in ₹ lakhs, e.g. 12 for 12,00,000): ")
        annual_salary_lakhs = float(salary_input)
        family_size_input = input("Enter family size (default 4): ") or "4"
        family_size = int(family_size_input)
    except ValueError:
        print("Invalid input, using example values: salary=12 lakhs, family size=4")
        annual_salary_lakhs, family_size = 12, 4

    print_report(annual_salary_lakhs, family_size)
