import subprocess
import os

print("\n🚀 Running Full GDP Forecast Integration Test Pipeline...\n")

# === Paths to individual modules ===
scripts = {
    "Backtest": r"D:\Projects\GDP\testing\scripts\backtest.py",
    "Evaluation": r"D:\Projects\GDP\testing\scripts\evaluation.py",
    "SHAP Analysis": r"D:\Projects\GDP\testing\scripts\shap_analysis.py"
}

# === Execute each script ===
for label, path in scripts.items():
    print(f"\n🟡 Running {label} Module...")
    try:
        subprocess.run(["python", path], check=True)
        print(f"✅ {label} executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {label} failed. Error: {e}")
        exit(1)

# === Output file validation ===
expected_outputs = [
    r"D:\Projects\GDP\testing\backtest_2020_2024.csv",
    r"D:\Projects\GDP\results\national\plots\gdp_forecast_comparison.png",
    r"D:\Projects\GDP\results\national\plots\sarimax_residual_histogram.png",
    r"D:\Projects\GDP\results\national\plots\shap_summary_unified.png"
]

print("\n📁 Verifying output files...")

missing = []
for fpath in expected_outputs:
    if not os.path.exists(fpath):
        print(f"❌ Missing: {fpath}")
        missing.append(fpath)
    else:
        print(f"✅ Found: {fpath}")

# === Final status ===
if missing:
    print("\n❌ Integration Test FAILED — Some outputs are missing.")
else:
    print("\n✅ Integration Test PASSED — All outputs successfully generated.")

print("\n📦 Integration Testing Complete.")
