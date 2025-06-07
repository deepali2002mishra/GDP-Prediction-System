import pandas as pd
import xgboost as xgb
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# === Load Data ===
df = pd.read_csv("data/processed/processed_data.csv")
sarimax_df = pd.read_csv("data/processed/sarimax_predictions.csv")

# === Merge SARIMAX predictions into dataset ===
df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")

# === Compute residuals: Actual - SARIMAX ===
df["Residual"] = df["GDP Growth (%)"] - df["SARIMAX_Pred"]

# === Drop rows with missing values (usually future years) ===
df.dropna(subset=["Residual"], inplace=True)

# === Inject event-aware flags (optional but helpful) ===
df["Crisis_2020"] = (df["Year"] == 2020).astype(int)
df["Recovery_2021_2022"] = df["Year"].isin([2021, 2022]).astype(int)
df["Policy_Push_2023"] = (df["Year"] == 2023).astype(int)

# === Drop unused or harmful columns ===
exclude_cols = ["Year", "GDP Growth (%)", "SARIMAX_Pred", "Residual"]

# === Select numerical features only, excluding target columns ===
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

print("🧠 Final Features Used for Training:")
print(feature_cols)

X = df[feature_cols]
y = df["Residual"]

# === Create output dir ===
os.makedirs("models", exist_ok=True)

# === Time Series Split for evaluation ===
tscv = TimeSeriesSplit(n_splits=3)
rmse_scores = []
results_df = pd.DataFrame()

plt.figure(figsize=(14, 6))
plt.title("📈 Residual Predictions vs Actual (All Folds)")

for i, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.025,
        "max_depth": 4,
        "lambda": 2.0,
        "alpha": 1.0,
        "eval_metric": "rmse"
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=350,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=20,
        verbose_eval=False
    )

    y_pred = model.predict(dval)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(rmse)
    print(f"📉 Fold {i+1} RMSE: {rmse:.3f}")

    fold_result = pd.DataFrame({
        "Fold": i + 1,
        "Index": val_idx,
        "Year": df.iloc[val_idx]["Year"].values,
        "True_Residual": y_val.values,
        "Predicted_Residual": y_pred
    })

    results_df = pd.concat([results_df, fold_result], ignore_index=True)

    plt.plot(df.iloc[val_idx]["Year"], y_val, label=f"Actual Fold {i+1}", linestyle='--')
    plt.plot(df.iloc[val_idx]["Year"], y_pred, label=f"Predicted Fold {i+1}")

# === Report Average RMSE ===
print(f"\n✅ Residual Model Average RMSE: {np.mean(rmse_scores):.3f}")

# === Save model ===
model.save_model("models/xgb_residual.json")
print("📦 Model saved to models/xgb_residual.json")

# === Save predictions to CSV ===
results_df.to_csv("data/processed/xgb_residual_predictions.csv", index=False)
print("📝 Residual predictions saved to models/xgb_residual_predictions.csv")

# === Save plot ===
plt.xlabel("Year")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/national/plots/xgb_residual_plot.png", dpi=300)
print("📊 Residual prediction plot saved to models/xgb_residual_plot.png")