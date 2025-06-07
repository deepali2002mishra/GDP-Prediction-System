import pandas as pd
import os

# === FILE PATHS ===
cleaned_path = "data/processed/cleaned_data.csv"
processed_path = "data/processed/processed_data.csv"

# Sectoral Raw Data
agri_raw_path = "data/raw/crop_export_production_stable.csv"
climate_raw_path = "data/raw/india_climate_soil_1961_2017.csv"
it_raw_path = "data/raw/IT_Sector_India_2010_2020.csv"

# === CONFIG: expected engineered columns for national processed data ===
expected_columns = [
    "Inflation Rate (%)_lag2",
    "Bank Credit Growth (%)_lag1",
    "Exports (Billion USD)_lag1",
    "Fiscal Deficit (% of GDP)_lag1",
    "FDI (Billion USD)_lag1"
]

# === VALIDATION FUNCTION ===
def validate_file(path, file_label, year_col="Year", expected_cols=None):
    print(f"\n🔍 Validating {file_label}: {path}")
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded {file_label} successfully. Shape: {df.shape}")

        if df.isnull().values.any():
            print(f"❌ ERROR: {file_label} has missing values.")
        else:
            print(f"✅ No missing values in {file_label}.")

        if expected_cols:
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ ERROR: {file_label} missing columns: {missing_cols}")
            else:
                print(f"✅ All expected columns found in {file_label}.")
        else:
            print("ℹ️ Column check not enforced for this dataset.")

        if year_col in df.columns:
            if not df[year_col].between(1960, 2035).all():
                print(f"❌ ERROR: {file_label} has year values out of range.")
            else:
                print(f"✅ All years in expected range for {file_label}.")
        else:
            print(f"⚠️ WARNING: {file_label} missing '{year_col}' column.")

    except Exception as e:
        print(f"❌ Failed to load {file_label}: {e}")

# === RUN VALIDATIONS ===
validate_file(cleaned_path, "Cleaned Data")
validate_file(processed_path, "Processed Data", expected_cols=expected_columns)

# === SECTORAL VALIDATIONS ===
print("\n📂 Sectoral Data Validation")
validate_file(agri_raw_path, "Agriculture Exports & Production (Raw)")
validate_file(climate_raw_path, "Climate & Soil Data (Raw)")
validate_file(it_raw_path, "IT Sector Indicators (Raw)")
