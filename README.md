# 📈 GDP and Sectoral Growth Forecasting using Economic Indicators
Capstone Project | B.Tech Computer Science | VIT University
Contributors: Sharvadharshi K, Swagatam Bera, Deepali Mishra
Supervisor: Dr. Suganthini C

## 🧠 Overview
In a dynamic economic environment, traditional GDP forecasting falls short by ignoring external economic factors. This project presents a hybrid forecasting framework using SARIMAX + XGBoost to accurately predict India’s national GDP and state-wise sectoral growth (Agriculture and IT), based on exogenous macroeconomic indicators.
A Streamlit-powered dashboard brings all insights to life, providing interactive visualizations and data-driven investment recommendations for stakeholders such as policymakers, investors, and researchers.

## 🚀 Features
📊 National GDP Forecasting using SARIMAX with exogenous inputs (e.g., CPI, interest rate, exports)

🌾 State-wise Agricultural Forecasting using rainfall, soil pH, crop exports, and prices

💻 IT Sector Revenue Forecasting using FDI, unemployment, global indices, and repo rates

📉 Hybrid Model: SARIMAX + XGBoost residual learning

🧪 Backtesting & Forecast Evaluation using RMSE, MAE, MAPE, and R² metrics

📈 Scenario-Based Forecasting: Baseline, Reform, Crisis, Mixed

🧭 Investment Strategy Output: Identifies high-growth states/sectors

🌐 Interactive Streamlit Dashboard for dynamic, real-time data exploration

## 🧰 Tech Stack
Python 3.9+	Core development

Pandas, NumPy	Data preprocessing and transformation

Statsmodels (SARIMAX)	Time-series forecasting

XGBoost	Machine learning for residual correction

Matplotlib, Seaborn	Visualization

Streamlit	Web app/dashboard

Git + GitHub	Version control

Jupyter/VS Code	Development environment


## 📊 Forecasting Methodology
Data Preprocessing: Merge, clean, align macroeconomic indicators and sector-specific datasets (1980–2024).

Feature Engineering: Lag features, rolling averages, log transforms, differencing.

SARIMAX Modeling: Forecast target variable (e.g., GDP) using exogenous inputs.

XGBoost Residual Modeling: Train on SARIMAX residuals to capture nonlinear trends.

Final Forecast: Hybrid = SARIMAX + XGBoost Residual

Dashboard Visualization: Render interactive projections with filters by state, sector, scenario.

## 📈 Performance

RMSE	0.848

MAE	0.720

MAPE	9.96%

R² Score	0.978

Directional Accuracy	100%


## 🖥️ Dashboard 
🔗 Run the dashboard using:
streamlit run dashboard/app.py

Key Dashboard Sections:
National GDP Projections (1980–2030)

Sectoral Forecasts: Agriculture & IT (state-wise)

Scenario Simulation: Reform, Crisis, Mixed

Investment Recommendations & Insights


## 📈 Sample Results
Scenario	Avg GDP Growth (2027–2030)

Reform Acceleration	7.88%

Mixed Recovery	7.34%

External Crisis	7.08%

Baseline (2025–26)	6.69%, 7.22%


## 📚 References
Government of India datasets (MoSPI, RBI, Agri Stats)

World Bank macroeconomic indicators

Research literature on SARIMAX, XGBoost, hybrid modeling

## 📌 Future Enhancements
Incorporate deep learning (LSTM) for long-range forecasting

Expand sectors (e.g., manufacturing, energy)

Integrate real-time APIs for data updates

Deploy dashboard online via Streamlit Cloud or Heroku


## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
