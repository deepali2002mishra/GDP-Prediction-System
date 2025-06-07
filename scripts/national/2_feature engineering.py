import pandas as pd

# === Load cleaned dataset ===
df = pd.read_csv("data/processed/cleaned_data.csv")
df = df.sort_values("Year")

# === Columns to apply feature engineering ===
macro_cols = [
    "GDP Growth (%)", "Inflation Rate (%)", "Interest Rate (%)",
    "Exchange Rate (USD/INR)", "Fiscal Deficit (% of GDP)",
    "Unemployment Rate (%)", "Money Supply (M3) Growth (%)",
    "Bank Credit Growth (%)", "Exports (Billion USD)",
    "Imports (Billion USD)", "FDI (Billion USD)"
]

# === Generate lag and rolling features ===
for col in macro_cols:
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_ma3"] = df[col].rolling(window=3).mean()

# === GDP-specific indicators ===
df["GDP_Trend_RollMean5"] = df["GDP Growth (%)"].rolling(window=5).mean()
df["GDP_Change_YoY"] = df["GDP Growth (%)"].diff()

# === Optional: Add policy reform boost flag manually ===
df["Reform_Policy_Boost"] = df["Year"].apply(lambda y: 1 if y in [2025, 2026] else 0)

# === Fill NA values (forward + backward) ===
df = df.fillna(method="bfill").fillna(method="ffill")

# === Save processed data ===
df.to_csv("data/processed/processed_data.csv", index=False)
print("✅ Feature engineering complete. Saved to data/processed_data.csv")
