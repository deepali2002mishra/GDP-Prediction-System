import pandas as pd
import numpy as np
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import os

# === Simulate future features with drift ===
def simulate_future_features(df, years, scenario_type="baseline"):
    last_row = df.iloc[-1].copy()
    second_last = df.iloc[-2].copy()
    ma3 = df["GDP Growth (%)"].tail(3).mean()
    simulated = []

    # Scenario-specific drift logic
    if scenario_type == "reform":
        drift_by_year = {
            2027: {"FDI (Billion USD)_lag1": 5.0, "Exports (Billion USD)_lag1": 8.0, "Fixed Capital Formation (% of GDP)": 0.8},
            2028: {"FDI (Billion USD)_lag1": 6.0, "Bank Credit Growth (%)_lag1": 1.2, "Reform_Policy_Boost": 1},
            2029: {"Exports (Billion USD)_lag1": 10.0, "Money Supply (M3) Growth (%)_lag1": 0.5},
            2030: {"GDP Growth (%)_lag1": 0.5, "Bank Credit Growth (%)_lag1": 1.0, "Fixed Capital Formation (% of GDP)": 1.0}
        }
    elif scenario_type == "crisis":
        drift_by_year = {
            2027: {"Inflation Rate (%)_lag2": 1.2, "Unemployment Rate (%)_lag1": 1.0, "FDI (Billion USD)_lag1": -2.0},
            2028: {"Exports (Billion USD)_lag1": -5.0, "Bank Credit Growth (%)_lag1": -1.5},
            2029: {"GDP Growth (%)_lag1": -0.6, "Reform_Policy_Boost": -1},
            2030: {"Money Supply (M3) Growth (%)_lag1": -0.8, "Interest Rate (%)_lag1": 1.5}
        }
    elif scenario_type == "mixed":
        drift_by_year = {
            2027: {"Inflation Rate (%)_lag2": 1.0, "Exports (Billion USD)_lag1": -3.0},
            2028: {"GDP Growth (%)_lag1": -0.3, "Bank Credit Growth (%)_lag1": -1.0},
            2029: {"FDI (Billion USD)_lag1": 3.0, "Reform_Policy_Boost": 0.5, "Fixed Capital Formation (% of GDP)": 0.5},
            2030: {"Exports (Billion USD)_lag1": 5.0, "GDP Growth (%)_lag1": 0.4}
        }
    else:
        # Default baseline
        drift_by_year = {
            2027: {"Inflation Rate (%)_lag2": 0.2, "Bank Credit Growth (%)_lag1": 0.5, "GDP Growth (%)_lag1": -0.2},
            2028: {"Inflation Rate (%)_lag2": -0.1, "FDI (Billion USD)_lag1": 2.0, "GDP Growth (%)_lag1": 0.1},
            2029: {"Interest Rate (%)_lag1": -0.1, "Exports (Billion USD)_lag1": 3.0},
            2030: {"Money Supply (M3) Growth (%)_lag1": 0.3, "Fixed Capital Formation (% of GDP)": 0.5}
        }

    for year in years:
        row = last_row.copy()
        row["Year"] = year

        # Apply drift
        drift = drift_by_year.get(year, {})
        for col, delta in drift.items():
            if col in row:
                row[col] += delta
            else:
                row[col] = delta

        # Update lags
        row["GDP Growth (%)_lag1"] = last_row["GDP Growth (%)"]
        row["GDP Growth (%)_lag2"] = second_last["GDP Growth (%)"]
        row["GDP Growth (%)_ma3"] = np.mean([
            row["GDP Growth (%)_lag1"],
            row["GDP Growth (%)_lag2"],
            ma3
        ])

        # Ensure Reform_Boost exists
        if "Reform_Policy_Boost" not in row:
            row["Reform_Policy_Boost"] = 0

        # Update history
        second_last = last_row.copy()
        last_row = row.copy()

        simulated.append(row)

    return pd.DataFrame(simulated)

# === Forecast GDP using SARIMAX + XGBoost ===
def forecast_gdp(future_df, xgb_model, sarimax_model, exog_cols, feature_cols, filename):
    sarimax_forecast = sarimax_model.get_forecast(steps=len(future_df), exog=future_df[exog_cols])
    future_df["SARIMAX_Pred"] = sarimax_forecast.predicted_mean.values

    dmatrix = xgb.DMatrix(future_df[feature_cols])
    correction = xgb_model.predict(dmatrix)
    correction = np.clip(correction, -1.0, 1.0)

    future_df["Final GDP Forecast (%)"] = future_df["SARIMAX_Pred"] + correction
    future_df[["Year", "SARIMAX_Pred", "Final GDP Forecast (%)"]].to_csv(filename, index=False)
    print(f"✅ Forecast saved to {filename}")

# === Main Execution ===
if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_data.csv")
    sarimax_model = SARIMAXResults.load("models/sarimax_gdp_model.pkl")
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgb_residual.json")

    exclude_cols = ["Year", "GDP Growth (%)", "SARIMAX_Pred", "Residual"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

    exog_cols = [
        "Inflation Rate (%)_lag2", "Fiscal Deficit (% of GDP)_lag1", "Interest Rate (%)_lag1",
        "Money Supply (M3) Growth (%)_lag1", "Exchange Rate (USD/INR)_lag1",
        "Unemployment Rate (%)_lag1", "Bank Credit Growth (%)_lag1",
        "FDI (Billion USD)_lag1", "Exports (Billion USD)_lag1", "Fixed Capital Formation (% of GDP)"
    ]

    os.makedirs("results/national", exist_ok=True)

    # Baseline Forecast (2025–26)
    baseline = simulate_future_features(df, [2025, 2026], scenario_type="baseline")
    forecast_gdp(baseline, xgb_model, sarimax_model, exog_cols, feature_cols,
                 "results/national/gdp_forecast_baseline_2025_2026.csv")

    # Scenario Forecasts (2027–2030)
    years = [2027, 2028, 2029, 2030]

    reform = simulate_future_features(df, years, scenario_type="reform")
    forecast_gdp(reform, xgb_model, sarimax_model, exog_cols, feature_cols,
                 "results/national/gdp_forecast_reform_2027_2030.csv")

    crisis = simulate_future_features(df, years, scenario_type="crisis")
    forecast_gdp(crisis, xgb_model, sarimax_model, exog_cols, feature_cols,
                 "results/national/gdp_forecast_crisis_2027_2030.csv")

    mixed = simulate_future_features(df, years, scenario_type="mixed")
    forecast_gdp(mixed, xgb_model, sarimax_model, exog_cols, feature_cols,
                 "results/national/gdp_forecast_mixed_2027_2030.csv")
