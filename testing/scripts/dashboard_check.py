import os

print("\n🔍 Streamlit Dashboard Sanity Check (Extended)\n")

# === Helper Function ===
def check_files(label, file_list):
    print(f"\n📁 Checking {label}...")
    missing = False
    for path in file_list:
        if not os.path.exists(path):
            print(f"❌ MISSING: {path}")
            missing = True
        else:
            print(f"✅ Found:   {path}")
    return not missing

# === 1. National Forecast Files ===
national_csvs = [
    "results/national/gdp_forecast_baseline_2025_2026.csv",
    "results/national/gdp_forecast_reform_2027_2030.csv",
    "results/national/gdp_forecast_crisis_2027_2030.csv",
    "results/national/gdp_forecast_mixed_2027_2030.csv",
    "results/national/scenario_summary_report.txt",
    "testing/backtest_2020_2024.csv"
]
national_plots = [
    "results/national/plots/gdp_forecast_scenarios.png",
    "results/national/plots/gdp_forecast_comparison.png",
    "results/national/plots/sarimax_residual_histogram.png",
    "results/national/plots/model_evaluation_bar.png",
    "results/national/plots/feature_correlation_top25.png",
    "results/national/plots/actual_gdp_growth_1980_2030.png",
    "results/national/plots/shap_summary_unified.png"
]

# === 2. Agriculture Forecast Files ===
agri_plot_dir = "results/sectoral/agriculture/plots"
agri_report_dir = "results/sectoral/agriculture/reports"

agri_plot_files = sorted([
    os.path.join(agri_plot_dir, fname) for fname in os.listdir(agri_plot_dir)
    if fname.endswith(".png")
])
agri_report_files = sorted([
    os.path.join(agri_report_dir, fname) for fname in os.listdir(agri_report_dir)
    if fname.endswith(".txt")
])

# === 3. IT Sector Forecast Files ===
it_plot_dir = "results/sectoral/IT/plots"
it_report_dir = "results/sectoral/IT/reports"

it_plot_files = [
    os.path.join(it_plot_dir, "combined_forecast_trends.png"),
    os.path.join(it_plot_dir, "top3_growth_bar_chart.png")
]
it_report_files = [
    os.path.join(it_report_dir, "top3_investment_strategy.txt")
]

# === Run All Checks ===
status_national_csvs = check_files("National Forecast CSVs", national_csvs)
status_national_plots = check_files("National Forecast Plots", national_plots)
status_agri_plots = check_files("Agriculture Forecast Plots", agri_plot_files)
status_agri_reports = check_files("Agriculture Reports", agri_report_files)
status_it_plots = check_files("IT Sector Plots", it_plot_files)
status_it_reports = check_files("IT Sector Report", it_report_files)

# === Final Verdict ===
if all([status_national_csvs, status_national_plots, status_agri_plots, status_agri_reports, status_it_plots, status_it_reports]):
    print("\n✅ Streamlit Dashboard Check PASSED: All required assets are present.")
else:
    print("\n❌ Streamlit Dashboard Check FAILED: Some files are missing.")
