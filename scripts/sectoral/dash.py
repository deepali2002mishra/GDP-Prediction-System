import streamlit as st
import os
from PIL import Image

# === Base Paths ===
AGRI_PLOTS_PATH = r"D:/Projects/GDP/results/sectoral/agriculture/plots"
AGRI_REPORTS_PATH = r"D:/Projects/GDP/results/sectoral/agriculture/reports"
IT_PLOTS_PATH = r"D:/Projects/GDP/results/sectoral/IT/plots"
IT_REPORTS_PATH = r"D:/Projects/GDP/results/sectoral/IT/reports"

# === Streamlit Config ===
st.set_page_config(page_title="India GDP Sectoral Forecast Dashboard", layout="wide")
st.title("📊 India GDP Sectoral Forecast Dashboard")

# === Define Tabs ===
agri_tab, it_tab = st.tabs(["🌾 Agriculture", "💻 IT Sector"])

# -------------------- AGRICULTURE TAB --------------------
with agri_tab:
    st.header("🌾 Agriculture Sector Forecast")

    # Tabs under Agriculture
    state_tab, top5_tab, national_tab = st.tabs(["📄 State Report", "🏆 Top 5 Crops Across India", "🌐 National Summary"])

    # ---------------- STATE REPORT TAB ----------------
    with state_tab:
        # Extract state names
        states = sorted([
            f.replace("_report.txt", "").replace("_", " ")
            for f in os.listdir(AGRI_REPORTS_PATH)
            if f.endswith("_report.txt") and not f.startswith("national")
        ])
        selected_state = st.selectbox("Select State", states, key="agri_state")

        # Load selected state report
        filename = f"{selected_state.replace(' ', '_')}_report.txt"
        report_path = os.path.join(AGRI_REPORTS_PATH, filename)

        st.subheader(f"📄 Forecast Report: {selected_state}")
        try:
            with open(report_path, "r", encoding='utf-8') as file:
                report_text = file.read()
                st.text_area("Agriculture Report", report_text, height=400)
        except FileNotFoundError:
            st.error(f"Report not found for {selected_state}")

        # Show related forecast plots
        plot_files = [
            f for f in os.listdir(AGRI_PLOTS_PATH)
            if f.startswith(selected_state.replace(" ", "_"))
        ]
        for plot_file in plot_files:
            st.image(Image.open(os.path.join(AGRI_PLOTS_PATH, plot_file)), caption=plot_file, use_column_width=True)

    # ---------------- TOP 5 CROPS TAB ----------------
    with top5_tab:
        st.subheader("🏆 Top 5 Agricultural Investment Opportunities Across India")
        nat_chart = os.path.join(AGRI_PLOTS_PATH, "national_top_5_bar_chart.png")
        if os.path.exists(nat_chart):
            st.image(Image.open(nat_chart), caption="Top 5 Crops by Investment Score", use_column_width=True)
        else:
            st.warning("Top 5 crop chart not found.")

    # ---------------- NATIONAL SUMMARY TAB ----------------
    with national_tab:
        st.subheader("🌐 Data-Driven Investment Rationale")
        national_report_path = os.path.join(AGRI_REPORTS_PATH, "national_top_5_report.txt")
        if os.path.exists(national_report_path):
            with open(national_report_path, "r", encoding='utf-8') as file:
                national_text = file.read()
                st.text_area("National Investment Summary & Rationale", national_text, height=500)
        else:
            st.warning("National summary report not found.")

# -------------------- IT TAB --------------------
with it_tab:
    st.header("💻 IT Sector Forecast")

    it_trend_path = os.path.join(IT_PLOTS_PATH, "combined_forecast_trends.png")
    it_top3_path = os.path.join(IT_PLOTS_PATH, "top3_growth_bar_chart.png")
    strategy_path = os.path.join(IT_REPORTS_PATH, "top3_investment_strategy.txt")

    # Display both plots side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 IT Revenue Trends")
        if os.path.exists(it_trend_path):
            st.image(Image.open(it_trend_path), use_column_width=True)
        else:
            st.warning("Revenue trend plot not found.")

    with col2:
        st.subheader("📊 Top 3 States by Growth")
        if os.path.exists(it_top3_path):
            st.image(Image.open(it_top3_path), use_column_width=True)
        else:
            st.warning("Top 3 growth chart not found.")

    # Investment strategy text
    st.subheader("📃 Strategic Investment Plan for Top 3 States")
    if os.path.exists(strategy_path):
        with open(strategy_path, "r", encoding='utf-8') as file:
            strategy_text = file.read()
            st.text_area("Investment Strategy", strategy_text, height=400)
    else:
        st.warning("Strategy report not found.")

# -------------------- Footer --------------------
st.markdown("---")

