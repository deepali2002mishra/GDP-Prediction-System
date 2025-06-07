import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# === Load datasets ===
df = pd.read_csv("data/processed/processed_data.csv")
sarimax_df = pd.read_csv("data/processed/sarimax_predictions.csv")
baseline_df = pd.read_csv("results/national/gdp_forecast_baseline_2025_2026.csv")
reform_df = pd.read_csv("results/national/gdp_forecast_reform_2027_2030.csv")
crisis_df = pd.read_csv("results/national/gdp_forecast_crisis_2027_2030.csv")
mixed_df = pd.read_csv("results/national/gdp_forecast_mixed_2027_2030.csv")

# Assign scenario labels
baseline_df["Scenario"] = "Baseline"
reform_df["Scenario"] = "Reform"
crisis_df["Scenario"] = "Crisis"
mixed_df["Scenario"] = "Mixed"
scenario_df = pd.concat([reform_df, crisis_df, mixed_df], ignore_index=True)

# Correct GDP contraction for COVID year
df.loc[df["Year"] == 2020, "GDP Growth (%)"] = -7.3

# === Merge SARIMAX predictions and compute residuals ===
df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")
df["Residual"] = df["GDP Growth (%)"] - df["SARIMAX_Pred"]

# === Evaluation metrics: SARIMAX (1980–2024) ===
actual = df["GDP Growth (%)"]
sarimax_pred = df["SARIMAX_Pred"]
sarimax_rmse = mean_squared_error(actual, sarimax_pred) ** 0.5
sarimax_mae = mean_absolute_error(actual, sarimax_pred)
sarimax_mape = np.mean(np.abs((actual - sarimax_pred) / actual)) * 100
sarimax_r2 = r2_score(actual, sarimax_pred)

# === Final Forecast (2025–2030) ===
forecast_df = pd.concat([baseline_df, scenario_df], ignore_index=True)
final_forecast = forecast_df["Final GDP Forecast (%)"]
final_baseline = forecast_df["SARIMAX_Pred"]

# === Evaluation metrics: Hybrid (2025–30) ===
hybrid_rmse = mean_squared_error(final_baseline, final_forecast) ** 0.5
hybrid_mae = mean_absolute_error(final_baseline, final_forecast)
hybrid_mape = np.mean(np.abs((final_forecast - final_baseline) / final_baseline)) * 100

# === Output directory ===
os.makedirs("plots", exist_ok=True)

# === Plot 1: GDP Forecasts by Scenario (2025–2030) ===
plt.figure(figsize=(12, 6))
for scenario in forecast_df["Scenario"].dropna().unique():
    subset = forecast_df[forecast_df["Scenario"] == scenario]
    plt.plot(subset["Year"], subset["Final GDP Forecast (%)"], marker='o', label=scenario)
plt.title("GDP Forecasts by Scenario (2025–2030)")
plt.xlabel("Year")
plt.ylabel("Final GDP Forecast (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/national/plots/gdp_forecast_scenarios.png", bbox_inches='tight')
plt.close()

# === Plot 2: Actual vs Final Forecast (1980–2030) ===
plt.figure(figsize=(12, 6))

# Plot actual GDP (1980–2024)
plt.plot(df["Year"], df["GDP Growth (%)"], label="Actual GDP (1980–2024)", color='black', linewidth=2)

# Plot each scenario's final forecast (2025–2030)
scenario_colors = {
    "Baseline": "green",
    "Reform": "orange",
    "Crisis": "red",
    "Mixed": "purple"
}

for scenario in forecast_df["Scenario"].dropna().unique():
    subset = forecast_df[forecast_df["Scenario"] == scenario]
    plt.plot(
        subset["Year"],
        subset["Final GDP Forecast (%)"],
        label=f"{scenario} Forecast",
        linestyle='--',
        marker='o',
        color=scenario_colors.get(scenario, 'gray')
    )

plt.xlabel("Year")
plt.ylabel("GDP Growth (%)")
plt.title("Actual vs GDP Forecast (1980–2030)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/national/plots/gdp_forecast_comparison.png", bbox_inches='tight')
plt.close()



# === Plot 3: SARIMAX Residual Histogram ===
plt.figure(figsize=(8, 5))
sns.histplot(df["Residual"], kde=True, color="skyblue", bins=20)
plt.title("SARIMAX Residual Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/national/plots/sarimax_residual_histogram.png", bbox_inches='tight')
plt.close()

# === Plot 4: Top 25 Correlation Matrix ===
exclude_cols = ["Year", "GDP Growth (%)", "SARIMAX_Pred", "Residual"]
numeric_cols = df.drop(columns=exclude_cols, errors='ignore').select_dtypes(include=[np.number])
corr = numeric_cols.corr()
top_corr = corr.iloc[:25, :25]

plt.figure(figsize=(16, 12))
sns.heatmap(top_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True,
            annot_kws={"size": 8}, cbar_kws={"shrink": 0.7})
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.title("Top 25 Correlated Macroeconomic Features", fontsize=14)
plt.tight_layout()
plt.savefig("results/national/plots/feature_correlation_top25.png", dpi=300, bbox_inches='tight')
plt.close()

# === Plot 5: Model Evaluation Bar Chart ===
metrics = ['RMSE', 'MAE', 'MAPE']
sarimax_vals = [sarimax_rmse, sarimax_mae, sarimax_mape]
hybrid_vals = [hybrid_rmse, hybrid_mae, hybrid_mape]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, sarimax_vals, width, label='SARIMAX', color='skyblue')
plt.bar(x + width/2, hybrid_vals, width, label='Hybrid', color='seagreen')
plt.xticks(x, metrics)
plt.ylabel("Error")
plt.title("Model Evaluation: SARIMAX vs Hybrid")
plt.legend()
plt.tight_layout()
plt.savefig("results/national/plots/model_evaluation_bar.png", bbox_inches='tight')
plt.close()

# === Plot 6: Actual GDP Growth (1980–2030) ===
actual_years = df[["Year", "GDP Growth (%)"]].copy()
actual_years.rename(columns={"GDP Growth (%)": "GDP Growth"}, inplace=True)
actual_years.loc[actual_years["Year"] == 2020, "GDP Growth"] = -7.3

plt.figure(figsize=(12, 6))
plt.plot(actual_years["Year"], actual_years["GDP Growth"], label="Actual GDP", color='black')
plt.xlabel("Year")
plt.ylabel("GDP Growth (%)")
plt.title("Actual GDP Growth (1980–2030)")
plt.grid(True)
plt.legend()
plt.ylim(-6.5, 10)
plt.tight_layout()
plt.savefig("results/national/plots/actual_gdp_growth_1980_2030.png", bbox_inches='tight')
plt.close()

# === Print Final Evaluation Summary ===
print("\n📊 MODEL EVALUATION SUMMARY\n")

print("🔷 SARIMAX Performance (1980–2024):")
print(f"   RMSE  : {sarimax_rmse:.3f}")
print(f"   MAE   : {sarimax_mae:.3f}")
print(f"   MAPE  : {sarimax_mape:.2f}%")
print(f"   R²    : {sarimax_r2:.3f}")

print("\n🔷 Hybrid Forecast (SARIMAX + XGBoost) Performance (2025–2030):")
print(f"   RMSE  : {hybrid_rmse:.3f}")
print(f"   MAE   : {hybrid_mae:.3f}")
print(f"   MAPE  : {hybrid_mape:.2f}%")

print("\n📈 Plots saved to 'plots/' directory:")
print("- gdp_forecast_scenarios.png")
print("- gdp_forecast_comparison.png")
print("- sarimax_residual_histogram.png")
print("- feature_correlation_top25.png")
print("- model_evaluation_bar.png")
print("- actual_gdp_growth_1980_2030.png")
