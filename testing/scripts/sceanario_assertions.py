import pandas as pd
import os

# === Load all forecast files ===
baseline = pd.read_csv("results/national/gdp_forecast_baseline_2025_2026.csv")
reform = pd.read_csv("results/national/gdp_forecast_reform_2027_2030.csv")
crisis = pd.read_csv("results/national/gdp_forecast_crisis_2027_2030.csv")
mixed = pd.read_csv("results/national/gdp_forecast_mixed_2027_2030.csv")

# === Combine & tag scenarios ===
baseline["Scenario"] = "Baseline"
reform["Scenario"] = "Reform"
crisis["Scenario"] = "Crisis"
mixed["Scenario"] = "Mixed"

forecast_df = pd.concat([baseline, reform, crisis, mixed], ignore_index=True)

# === Assertions ===

baseline_avg = baseline["Final GDP Forecast (%)"].mean()
reform_avg = reform["Final GDP Forecast (%)"].mean()
crisis_avg = crisis["Final GDP Forecast (%)"].mean()
mixed_avg = mixed["Final GDP Forecast (%)"].mean()

# 1. Reform > Baseline
assert reform_avg > baseline_avg, "❌ Reform scenario is not more optimistic than Baseline"

# 2. Crisis < Mixed
assert crisis_avg < mixed_avg, "❌ Crisis scenario is not worse than Mixed Recovery"

# 3. All values in realistic range
assert forecast_df["Final GDP Forecast (%)"].between(4.0, 10.0).all(), "❌ Forecasts exceed realistic GDP growth bounds"

# === Save summary report ===
summary_text = f"""
📊 GDP Forecast Scenario Summary (2025–2030)

✅ Reform Avg    : {reform_avg:.2f}%
✅ Baseline Avg  : {baseline_avg:.2f}%
✅ Mixed Avg     : {mixed_avg:.2f}%
✅ Crisis Avg    : {crisis_avg:.2f}%

Assertions:
✔ Reform > Baseline
✔ Crisis < Mixed
✔ All forecasts ∈ [4.0%, 10.0%]
"""

report_path = "results/national/scenario_summary_report.txt"
os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(summary_text.strip())

print(summary_text)
print(f"\n📝 Scenario summary report saved to: {report_path}")
