import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# === PAGE SETUP ===
st.set_page_config(page_title="India GDP Forecast Dashboard (2025–2030)", layout="wide")
st.title("India GDP Forecast Dashboard (2025–2030)")
st.markdown("Explore national GDP projections and sector-specific forecasts across Agriculture and IT.")

# === BASE PATH SETUP ===
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results", "national")
data_dir = os.path.join(base_dir, "data", "processed")

# === MAIN TABS ===
main_tab1, main_tab2 = st.tabs(["📊 National GDP", "📂 Sectoral GDP"])

# === NATIONAL GDP TAB ===
with main_tab1:
    st.header("National GDP Forecast (1980–2030)")

    def clean_columns(df):
        df.columns = df.columns.str.strip()
        return df

    def get_forecast_column(df):
        forecast_cols = [col for col in df.columns if "forecast" in col.lower()]
        if forecast_cols:
            return forecast_cols[0]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        return numeric_cols[-1]

    baseline = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_baseline_2025_2026.csv")))
    reform = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_reform_2027_2030.csv")))
    crisis = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_crisis_2027_2030.csv")))
    mixed = clean_columns(pd.read_csv(os.path.join(results_dir, "gdp_forecast_mixed_2027_2030.csv")))
    sarimax_df = pd.read_csv(os.path.join(data_dir, "sarimax_predictions.csv"))
    df = pd.read_csv(os.path.join(data_dir, "processed_data.csv"))

    for frame in [baseline, reform, crisis, mixed, sarimax_df, df]:
        frame["Year"] = frame["Year"].astype(int)

    baseline['Scenario'] = 'Baseline'
    reform['Scenario'] = 'Reform Acceleration'
    crisis['Scenario'] = 'External Crisis'
    mixed['Scenario'] = 'Mixed Recovery'

    forecast_df = pd.concat([baseline, reform, crisis, mixed], ignore_index=True)
    forecast_col = get_forecast_column(forecast_df)
    df = df.merge(sarimax_df[["Year", "SARIMAX_Pred"]], on="Year", how="left")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("📉 Historical vs Forecasted GDP (1980–2030)")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(df["Year"], df["GDP Growth (%)"], label="Actual GDP (1980–2024)", color='black', linewidth=2)
        scenario_colors = {
            "Baseline": "green",
            "Reform Acceleration": "orange",
            "External Crisis": "red",
            "Mixed Recovery": "purple"
        }
        for scenario in forecast_df["Scenario"].dropna().unique():
            subset = forecast_df[forecast_df["Scenario"] == scenario]
            ax1.plot(subset["Year"], subset[forecast_col], label=f"{scenario} Forecast", linestyle='--', marker='o', color=scenario_colors.get(scenario, 'gray'))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("GDP Growth (%)")
        ax1.set_title("Actual vs Forecast")
        ax1.grid(True)
        ax1.legend(fontsize=6)
        st.pyplot(fig1)

    with col2:
        st.subheader("📈 GDP Growth Projections by Scenario (2025–2030)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        for scenario, df_ in forecast_df.groupby('Scenario'):
            ax2.plot(df_["Year"], df_[forecast_col], marker='o', label=scenario)
        ax2.axhline(y=6.69, color='gray', linestyle='--', label='2025 Expected (6.69%)')
        ax2.set_ylabel("GDP Growth (%)")
        ax2.set_xlabel("Year")
        ax2.set_title("Scenario-Wise Projections")
        ax2.legend(fontsize=6)
        st.pyplot(fig2)

    st.markdown("### 🟢 Baseline Forecast Description")
    baseline_desc = baseline[["Year", forecast_col]].set_index("Year").T.to_dict()
    desc_text = "The **baseline GDP growth forecast** is as follows:\n\n"
    for year, value in baseline_desc.items():
        desc_text += f"- **{year}**: {value[forecast_col]:.2f}%\n"
    st.markdown(desc_text)

    st.subheader("📋 Scenario Forecast Comparison: 2027–2030 Only")
    gdp_2027 = forecast_df[(forecast_df["Year"] >= 2027) & (forecast_df["Scenario"] != 'Baseline')]
    table = gdp_2027.pivot(index='Year', columns='Scenario', values=forecast_col)
    table.index = table.index.astype(int).astype(str)
    st.dataframe(table.style.format("{:.2f}"), use_container_width=True)

    st.subheader("🧭 2025 Economic Recommendations")
    rec_path = os.path.join(results_dir, "recommendations_2025.txt")
    if os.path.exists(rec_path):
        with open(rec_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.warning("⚠️ Recommendations file not found.")

# === SECTORAL GDP TAB ===
with main_tab2:
    st.header("📂 Sectoral GDP Forecast (Agriculture & IT)")
    st.markdown("Explore forecasts for India's agriculture and IT sectors, including investment insights.")
    st.markdown("---")

    agri_tab, it_tab = st.tabs(["🌾 Agriculture Sector", "💻 IT Sector"])

    # === AGRICULTURE TAB ===
    with agri_tab:
        st.markdown("### 🌱 Agriculture Sector Forecast")
        st.info("Use the tabs below to view state-level forecasts or national investment opportunities.")
        state_tab, national_tab = st.tabs(["📍 State-wise Forecast", "🌐 National Investment Picks"])

        AGRI_PLOTS_PATH = os.path.join(base_dir, "results", "sectoral", "agriculture", "plots")
        AGRI_REPORTS_PATH = os.path.join(base_dir, "results", "sectoral", "agriculture", "reports")

        with state_tab:
            st.markdown("#### 📑 State Forecast Report")
            states = sorted([
                f.replace("_report.txt", "").replace("_", " ")
                for f in os.listdir(AGRI_REPORTS_PATH)
                if f.endswith("_report.txt") and not f.startswith("national")
            ])
            selected_state = st.selectbox("Select State", states, key="agri_state")
            st.markdown(f"##### 📊 Forecast Plot for {selected_state}")

            plot_files = [
                f for f in os.listdir(AGRI_PLOTS_PATH)
                if f.startswith(selected_state.replace(" ", "_"))
            ]
            for plot_file in plot_files:
                col_center = st.columns([1, 10, 1])[1]
                with col_center:
                    st.image(
                        Image.open(os.path.join(AGRI_PLOTS_PATH, plot_file)),
                        caption=plot_file,
                        width=1000
                    )

            st.markdown(f"##### 📄 Forecast Report for {selected_state}")
            filename = f"{selected_state.replace(' ', '_')}_report.txt"
            report_path = os.path.join(AGRI_REPORTS_PATH, filename)

            try:
                with open(report_path, "r", encoding='utf-8') as file:
                    report_text = file.read()
                    st.code(report_text, language="text")
            except FileNotFoundError:
                st.warning(f"Report not found for {selected_state}")

        with national_tab:
            st.markdown("#### 🌐 Investment Recommendations")
            national_report_path = os.path.join(AGRI_REPORTS_PATH, "national_top_5_report.txt")

            if os.path.exists(national_report_path):
                with open(national_report_path, "r", encoding='utf-8') as file:
                    national_text = file.read()

                # --- Parse rationale blocks ---
                sections = national_text.split("🏆 RECOMMENDATION")
                rationale_map = {}

                # Split using "🔍 Why invest in", skip the first empty split
                rationale_blocks = national_text.split("🔍 Why invest in")[1:]

                for block in rationale_blocks:
                    try:
                        title_line, *content_lines = block.strip().splitlines()
                        title = title_line.replace("?", "").strip().lower()  # e.g., 'jute in west bengal'
                        content = [line.strip('• ').strip() for line in content_lines if line.strip()]
                        rationale_map[title] = content
                    except Exception as e:
                        print(f"Error parsing rationale block: {block[:50]}...", e)

                # --- Render RECOMMENDATION blocks with their rationale ---
                for rec_section in sections[1:]:
                    lines = rec_section.strip().splitlines()
                    title_line = lines[0].strip()

                    # Crop/State extraction
                    crop, state = "UNKNOWN", "UNKNOWN"
                    try:
                        parts = title_line.split(":")[1].strip().split(" in ")
                        crop = parts[0].strip().lower()
                        state = parts[1].strip().lower()
                    except:
                        pass

                    lookup_key = f"{crop} in {state}".lower()

                    # Trim summary only until rationale starts
                    summary_lines = []
                    for line in lines[1:]:
                        if line.strip().startswith("🔍 Why invest in"):
                            break
                        summary_lines.append(line)
                    summary_part = "\n".join(summary_lines).strip()

                    with st.expander(f"📌 RECOMMENDATION {title_line}"):
                        st.markdown("##### 📊 Forecast Summary")
                        st.code(summary_part, language="text")

                        if lookup_key in rationale_map:
                            st.markdown("##### 💡 Investment Rationale")
                            st.markdown(f"🔎 **Why invest in {crop.title()} in {state.title()}?**")
                            for line in rationale_map[lookup_key]:
                                st.markdown(f"- {line}")
                        else:
                            st.info("ℹ️ No rationale available.")
            else:
                st.warning("National summary report not found.")



    # === IT TAB ===
    with it_tab:
        st.markdown("### 💻 IT Sector Forecast")
        st.info("View IT revenue trends and top growth states over the next 10 years.")

        IT_PLOTS_PATH = os.path.join(base_dir, "results", "sectoral", "IT", "plots")
        IT_REPORTS_PATH = os.path.join(base_dir, "results", "sectoral", "IT", "reports")

        trend_path = os.path.join(IT_PLOTS_PATH, "combined_forecast_trends.png")
        bar_path = os.path.join(IT_PLOTS_PATH, "top3_growth_bar_chart.png")
        strategy_path = os.path.join(IT_REPORTS_PATH, "top3_investment_strategy.txt")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 IT Revenue Trends")
            if os.path.exists(trend_path):
                st.image(Image.open(trend_path), use_column_width=True)
            else:
                st.warning("IT revenue trend chart not found.")

        with col2:
            st.markdown("#### 📊 Top 3 States by Growth")
            if os.path.exists(bar_path):
                st.image(Image.open(bar_path), use_column_width=True)
            else:
                st.warning("Top 3 IT growth chart not found.")

        st.markdown("#### 🧭 Strategic Investment Plan for Top 3 States")
        if os.path.exists(strategy_path):
            with open(strategy_path, "r", encoding='utf-8') as file:
                strategy_text = file.read()
                st.code(strategy_text, language="text")
        else:
            st.warning("Strategy report not found.")
