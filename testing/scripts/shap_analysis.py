import shap
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load processed data ===
df = pd.read_csv("data/processed/processed_data.csv")
sarimax_df = pd.read_csv("data/processed/sarimax_predictions.csv")
df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")
df["Residual"] = df["GDP Growth (%)"] - df["SARIMAX_Pred"]

# === Feature selection ===
exclude_cols = ["Year", "GDP Growth (%)", "SARIMAX_Pred", "Residual"]
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [float, int]]
X = df[feature_cols]

# === Load trained booster ===
booster = xgb.Booster()
booster.load_model("models/xgb_residual.json")

# === Create explainer
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X)

# === SHAP Summary Bar Plot ===
print("\n📊 SHAP Summary for Unified Residual Model:")
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Save the figure explicitly after rendering
plt.savefig(r"D:\Projects\GDP\results\national\plots\shap_summary_unified.png", bbox_inches='tight')
plt.close()
print("📈 SHAP summary plot saved to plots/shap_summary_unified.png")
