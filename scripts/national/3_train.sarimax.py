import pandas as pd
import statsmodels.api as sm
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_csv("data/processed/processed_data.csv")
df = df.dropna(subset=["GDP Growth (%)"])

# === Define target variable ===
y = df["GDP Growth (%)"]

# === Extended exogenous variables for realism ===
exog_cols = [
    "Inflation Rate (%)_lag2",
    "Fiscal Deficit (% of GDP)_lag1",
    "Interest Rate (%)_lag1",
    "Money Supply (M3) Growth (%)_lag1",
    "Exchange Rate (USD/INR)_lag1",
    "Unemployment Rate (%)_lag1",
    "Bank Credit Growth (%)_lag1",
    "FDI (Billion USD)_lag1",
    "Exports (Billion USD)_lag1",
    "Fixed Capital Formation (% of GDP)"
]
exog = df[exog_cols]

# === Fit SARIMAX model ===
model = sm.tsa.SARIMAX(
    y,
    exog=exog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 4),
    enforce_stationarity=False,
    enforce_invertibility=False
)
result = model.fit(disp=False)

# === Save model ===
os.makedirs("models", exist_ok=True)
result.save("models/sarimax_gdp_model.pkl")
print("✅ SARIMAX model saved.")

# === In-sample predictions ===
df["SARIMAX_Pred"] = result.predict(start=0, end=len(y)-1, exog=exog)
df[["Year", "GDP Growth (%)", "SARIMAX_Pred"]].to_csv("data/processed/sarimax_predictions.csv", index=False)
print("📈 In-sample predictions saved to data/processed/sarimax_predictions.csv")

# === Evaluate performance ===
rmse = mean_squared_error(df["GDP Growth (%)"], df["SARIMAX_Pred"]) ** 0.5
print(f"📉 SARIMAX RMSE: {rmse:.3f}")

# === Plot and save residuals ===
residuals = df["GDP Growth (%)"] - df["SARIMAX_Pred"]
plt.figure(figsize=(10, 5))
plt.plot(df["Year"], residuals, label="Residual")
plt.axhline(0, color='red', linestyle='--')
plt.title("SARIMAX Residuals Over Time")
plt.xlabel("Year")
plt.ylabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.legend()

# Save plot
os.makedirs("plots", exist_ok=True)
plot_path = "results/national/plots/sarimax_residuals.png"
plt.savefig(plot_path)
print(f"🖼️ Residual plot saved to {plot_path}")
