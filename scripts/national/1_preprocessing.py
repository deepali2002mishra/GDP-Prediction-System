import pandas as pd

# Load your dataset
df = pd.read_csv("data/raw/national_economic_indicators_1980_2024.csv")
print(f"Initial shape: {df.shape}")

# Strip column names and remove duplicates
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

# Remove any rows with invalid/missing Year or GDP Growth
df = df[df["Year"].notna() & df["GDP Growth (%)"].notna()]
df["Year"] = df["Year"].astype(int)

# Fix extreme outliers only for selected columns using IQR — non-cumulatively
def iqr_filter(col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f"{col}: removing {outliers} outlier(s)")
    df.loc[(df[col] < lower) | (df[col] > upper), col] = None  # replace with NaN

iqr_cols = [
    "GDP Growth (%)", "Inflation Rate (%)", "Interest Rate (%)", 
    "Fiscal Deficit (% of GDP)", "Unemployment Rate (%)"
]

for col in iqr_cols:
    if col in df.columns:
        iqr_filter(col)

# Interpolate missing numeric values (linear method), then forward/backward fill
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

# Final cleanup
df = df.sort_values("Year").reset_index(drop=True)
df.to_csv("data/processed/cleaned_data.csv", index=False)
print(f"✅ Cleaned dataset saved: data/cleaned_data.csv ({df.shape[0]} rows)")
