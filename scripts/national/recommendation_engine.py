import pandas as pd
import os

# === Load forecast data ===
forecast = pd.read_csv("results/national/gdp_forecast_baseline_2025_2026.csv")
row_2025 = forecast[forecast["Year"] == 2025].iloc[0]

# === Load supporting features (processed data for 2024 → lag predictors) ===
processed_df = pd.read_csv("data/processed/processed_data.csv")
lags_2025 = processed_df[processed_df["Year"] == 2024].iloc[0]

# === Extract relevant indicators ===
inflation = lags_2025["Inflation Rate (%)_lag2"]
interest_rate = lags_2025["Interest Rate (%)_lag1"]
unemployment = lags_2025["Unemployment Rate (%)_lag1"]
fiscal_deficit = lags_2025["Fiscal Deficit (% of GDP)_lag1"]
credit_growth = lags_2025["Bank Credit Growth (%)_lag1"]
fdi = lags_2025["FDI (Billion USD)_lag1"]
m3_growth = lags_2025["Money Supply (M3) Growth (%)_lag1"]
exports = lags_2025["Exports (Billion USD)_lag1"]
gdp_forecast = row_2025["Final GDP Forecast (%)"]

# === Initialize recommendation blocks ===
risks = []
opportunities = []
recommendations = []

# === Risk Analysis Based on Realistic Economic Thresholds ===
if inflation > 6:
    risks.append("High inflation may suppress real incomes and consumer confidence.")
if unemployment > 7:
    risks.append("Elevated unemployment levels could signal labor market stress.")
if interest_rate > 7.5:
    risks.append("Tight monetary policy may affect capital expenditure and borrowing.")
if fiscal_deficit > 6.5:
    risks.append("Rising fiscal deficit may lead to debt sustainability concerns.")
if gdp_forecast < 6:
    risks.append("Growth may not be strong enough to offset structural weaknesses.")

# === Opportunity Signals ===
if fdi > 60:
    opportunities.append("Strong FDI inflows indicate sustained global investor confidence in India.")
if credit_growth > 10:
    opportunities.append("High credit growth reflects strong business and consumer lending activity.")
if m3_growth > 8:
    opportunities.append("Ample liquidity suggests supportive monetary environment.")
if exports > 400:
    opportunities.append("Robust exports could buffer external imbalances.")
if gdp_forecast > 6.5:
    opportunities.append("Above-average GDP growth expected — India remains on a strong upward trajectory.")

# === Policy & Investor Recommendations ===
if inflation < 5 and credit_growth > 10:
    recommendations.append("Continue supportive credit policies to sustain expansion momentum.")
if fdi > 60:
    recommendations.append("Encourage long-term capital formation in infrastructure and technology.")
if fiscal_deficit > 6:
    recommendations.append("Tighten expenditure controls to maintain fiscal sustainability.")
if unemployment > 6:
    recommendations.append("Invest in labor-intensive sectors like manufacturing and construction.")

# === Format output ===
output_lines = []

output_lines.append("📅 Economic Recommendation Report: Year 2025\n")
output_lines.append(f"📈 GDP Forecast: {gdp_forecast:.2f}%\n")

output_lines.append("\n🔍 Key Risks:")
output_lines += [f"• {r}" for r in risks] if risks else ["• No significant macroeconomic risks projected."]

output_lines.append("\n💡 Economic Opportunities:")
output_lines += [f"• {o}" for o in opportunities] if opportunities else ["• No major opportunities identified."]

output_lines.append("\n🧭 Strategic Recommendations:")
output_lines += [f"• {rec}" for rec in recommendations] if recommendations else ["• Maintain current policy direction with caution."]

# === Save to file ===
os.makedirs("results", exist_ok=True)
with open("results/national/recommendations_2025.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

# === Print to console ===
print("\n".join(output_lines))
print("\n✅ Recommendation report")
