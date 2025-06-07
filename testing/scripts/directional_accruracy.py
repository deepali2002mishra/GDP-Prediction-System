import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("\n📊 Directional Accuracy Testing (2020–2024)\n")

# === Load backtest results ===
df = pd.read_csv("testing/backtest_2020_2024.csv")

# === Define direction: 1 if GDP growth is up, 0 if down
df["Actual_Direction"] = df["Actual_GDP_Growth"].diff().apply(lambda x: 1 if x > 0 else 0)
df["Predicted_Direction"] = df["Hybrid_Prediction"].diff().apply(lambda x: 1 if x > 0 else 0)

# Drop the first year (no previous to compare)
df = df.dropna().copy()

# === Metrics ===
acc = accuracy_score(df["Actual_Direction"], df["Predicted_Direction"])
prec = precision_score(df["Actual_Direction"], df["Predicted_Direction"])
rec = recall_score(df["Actual_Direction"], df["Predicted_Direction"])
f1 = f1_score(df["Actual_Direction"], df["Predicted_Direction"])
conf_matrix = confusion_matrix(df["Actual_Direction"], df["Predicted_Direction"])

# === Print Results ===
print("✅ Directional Classification Results (Hybrid GDP Forecast):\n")
print(f"📌 Accuracy : {acc:.3f}")
print(f"📌 Precision: {prec:.3f}")
print(f"📌 Recall   : {rec:.3f}")
print(f"📌 F1 Score : {f1:.3f}")
print("\n📊 Confusion Matrix [Actual vs Predicted]:")
print(conf_matrix)

# === Plot Confusion Matrix ===
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.xlabel('Predicted Direction')
plt.ylabel('Actual Direction')
plt.title('Confusion Matrix: GDP Direction Forecast (2020–2024)')
plt.tight_layout()

# Save the plot
conf_matrix_path = "testing/directional_confusion_matrix.png"
plt.savefig(conf_matrix_path, dpi=300)
plt.close()

print(f"\n🖼️ Confusion matrix plot saved to: {conf_matrix_path}")

# === Save Results Summary ===
summary = f"""
📊 Directional Accuracy Report (2020–2024)

✅ Accuracy : {acc:.3f}
✅ Precision: {prec:.3f}
✅ Recall   : {rec:.3f}
✅ F1 Score : {f1:.3f}

Confusion Matrix [Actual vs Predicted]:
{conf_matrix.tolist()}

Data Points Used: {len(df)}
"""

report_path = "testing/directional_accuracy_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(summary.strip())

print(f"\n📝 Saved report to: {report_path}")
