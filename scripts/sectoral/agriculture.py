import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define paths
BASE_PLOT_PATH = r"D:\Projects\GDP\results\sectoral\agriculture\plots"
BASE_REPORT_PATH = r"D:\Projects\GDP\results\sectoral\agriculture\reports"
os.makedirs(BASE_PLOT_PATH, exist_ok=True)
os.makedirs(BASE_REPORT_PATH, exist_ok=True)

# Load datasets
production_df = pd.read_csv(r"D:\Projects\GDP\data\raw\crop_export_production_stable.csv")
climate_df = pd.read_csv(r"D:\Projects\GDP\data\raw\india_climate_soil_1961_2017.csv")
merged_df = pd.merge(production_df, climate_df, on=['State', 'Year'], how='left')
merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
merged_df.dropna(subset=['Production Quantity', 'Export Volume', 'Annual Rainfall (mm)'], inplace=True)

def save_plot(fig, filename):
    fig.savefig(os.path.join(BASE_PLOT_PATH, filename), dpi=300)
    plt.close(fig)

def save_report(text, filename):
    with open(os.path.join(BASE_REPORT_PATH, filename), 'w', encoding='utf-8') as f:
        f.write(text)

def analyze_state(state_name, state_df):
    crop_results = []
    forecast_data = {}

    for crop in state_df['Crop'].unique():
        crop_df = state_df[state_df['Crop'] == crop].sort_values('Year')
        if len(crop_df) < 5:
            continue
        try:
            model = SARIMAX(
                endog=crop_df['Production Quantity'],
                exog=crop_df[['Export Volume', 'Annual Rainfall (mm)']],
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0)
            )
            model_fit = model.fit(disp=False)
            future_exog = pd.DataFrame({
                'Export Volume': [crop_df['Export Volume'].iloc[-3:].mean()] * 10,
                'Annual Rainfall (mm)': [crop_df['Annual Rainfall (mm)'].iloc[-3:].mean()] * 10
            })
            
            # Forecast next 10 years
            forecast = model_fit.get_forecast(steps=10, exog=future_exog)
            forecast_mean = forecast.predicted_mean
            conf_int = forecast.conf_int()
            
            # Calculate metrics
            avg_price = crop_df['Wholesale Price'].mean()
            growth_rate = (forecast_mean.mean() - crop_df['Production Quantity'].iloc[-1]) / crop_df['Production Quantity'].iloc[-1]
            price_volatility = crop_df['Wholesale Price'].std() / crop_df['Wholesale Price'].mean()
            export_dependence = crop_df['Export Volume'].mean() / crop_df['Production Quantity'].mean()
            rainfall_variability = crop_df['Annual Rainfall (mm)'].std() / crop_df['Annual Rainfall (mm)'].mean()
            
            # Soil analysis
            soil_ph = crop_df['Soil pH Level'].mean()
            organic_matter = crop_df['Organic Matter (%)'].mean()
            dominant_soil = crop_df['Soil Type'].mode()[0] if not crop_df['Soil Type'].mode().empty else "Unknown"
            soil_score = 0
            if 6 <= soil_ph <= 7:
                soil_score += 3
            elif 5.5 <= soil_ph < 6 or 7 < soil_ph <= 7.5:
                soil_score += 2
            if organic_matter >= 2:
                soil_score += 2
            elif 1 <= organic_matter < 2:
                soil_score += 1
            
            # Calculate final score
            base_score = forecast_mean.mean() * avg_price * (1 + growth_rate)
            adjusted_score = base_score * (1 + soil_score/5) * (1 - price_volatility/2)

            crop_results.append({
                'State': state_name,
                'Crop': crop,
                'Current_Production': crop_df['Production Quantity'].iloc[-1],
                'Forecasted_Production': forecast_mean.mean(),
                'Growth_Rate': growth_rate,
                'Avg_Price': avg_price,
                'Price_Volatility': price_volatility,
                'Export_Dependence': export_dependence,
                'Rainfall_Variability': rainfall_variability,
                'Soil_pH': soil_ph,
                'Organic_Matter': organic_matter,
                'Soil_Score': soil_score,
                'Score': adjusted_score
            })

            # Store forecast data for plotting
            forecast_data[crop] = {
                'history': crop_df['Production Quantity'],
                'forecast': forecast_mean,
                'conf_int': conf_int,
                'years': crop_df['Year'].tolist() + [max(crop_df['Year'])+i+1 for i in range(10)]
            }
        except Exception:
            continue

    if not crop_results:
        return None, None

    # Create results dataframe and rank crops
    results_df = pd.DataFrame(crop_results)
    results_df['State_Rank'] = results_df['Score'].rank(ascending=False)
    results_df = results_df.sort_values('State_Rank')
    
    return results_df, forecast_data

def plot_state_forecast(state_name, top_crop, forecast_data):
    data = forecast_data[top_crop['Crop']]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data['years'][:-10], data['history'], 'b-', label='Historical', linewidth=2)
    ax.plot(data['years'][-10:], data['forecast'], 'r--', label='10-Year Forecast', linewidth=2)
    ax.fill_between(data['years'][-10:], data['conf_int'].iloc[:, 0], data['conf_int'].iloc[:, 1],
                    color='pink', alpha=0.3, label='Confidence Interval')
    ax.set_title(f'{top_crop["Crop"]} Production Forecast in {state_name}', pad=20)
    ax.set_xlabel('Year')
    ax.set_ylabel('Production (tonnes)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()
    filename = f"{state_name}_{top_crop['Crop']}_forecast.png".replace(" ", "_")
    save_plot(fig, filename)

def print_state_report(state_name, state_results, forecast_data):
    report_lines = []
    
    report_lines.append(f"{'='*80}")
    report_lines.append(f"AGRICULTURAL ANALYSIS FOR {state_name.upper()} (10-YEAR FORECAST)")
    report_lines.append(f"{'='*80}")
    
    if state_results is None or state_results.empty:
        report_lines.append("\nNo valid crop data available for this state.")
        print("\n".join(report_lines))
        save_report("\n".join(report_lines), f"{state_name}_report.txt".replace(" ", "_"))
        return
    
    top_crop = state_results.iloc[0]
    report_lines.append(f"\n🏆 TOP CROP: {top_crop['Crop'].upper()}")
    report_lines.append(f"📍 Current Production: {top_crop['Current_Production']:,.2f} tonnes")
    report_lines.append(f"🔮 Forecasted Production (10yr avg): {top_crop['Forecasted_Production']:,.2f} tonnes")
    report_lines.append(f"📈 10-Year Growth Rate: {top_crop['Growth_Rate']:.2%}")
    report_lines.append(f"💰 Average Price: ₹{top_crop['Avg_Price']:,.2f}")
    report_lines.append(f"🌱 Soil Score: {top_crop['Soil_Score']}/5")
    report_lines.append(f"🏆 10-Year Investment Score: {top_crop['Score']:,.2f}")
    
    # Top 3 crops summary
    report_lines.append("\nTOP 3 CROPS IN THIS STATE:")
    top3 = state_results.head(3)
    for _, row in top3.iterrows():
        report_lines.append(
            f" - {row['Crop']}: Score={row['Score']:,.2f}, Growth={row['Growth_Rate']:.2%}, Price=₹{row['Avg_Price']:,.2f}"
        )

    # Print to console
    print("\n".join(report_lines))

    # Save to file
    filename = f"{state_name}_report.txt".replace(" ", "_")
    save_report("\n".join(report_lines), filename)

    # Save plot
    if top_crop['Crop'] in forecast_data:
        plot_state_forecast(state_name, top_crop, forecast_data)

def generate_report(state_name, top_crop, state_results):
    report_lines = [
        f"AGRICULTURAL REPORT FOR {state_name.upper()}",
        f"Top Crop: {top_crop['Crop']}",
        f"Current Production: {top_crop['Current_Production']:,.2f} tonnes",
        f"Forecasted (10yr Avg): {top_crop['Forecasted_Production']:,.2f} tonnes",
        f"Growth Rate: {top_crop['Growth_Rate']:.2%}",
        f"Avg Price: ₹{top_crop['Avg_Price']:,.2f}",
        f"Soil Score: {top_crop['Soil_Score']}/5",
        f"Investment Score: {top_crop['Score']:,.2f}",
        "\nTop 3 Crops:"
    ]
    top3 = state_results.head(3)
    for _, row in top3.iterrows():
        report_lines.append(f" - {row['Crop']}: Score={row['Score']:,.2f}, Growth={row['Growth_Rate']:.2%}, Price=₹{row['Avg_Price']:,.2f}")
    return "\n".join(report_lines)

def generate_investment_rationale(state, crop, merged_df, national_results):
    try:
        state_crop_data = merged_df[(merged_df['State'] == state) & (merged_df['Crop'] == crop)]
        if state_crop_data.empty:
            return [f"No data available for {crop} in {state}."]
        
        # Calculate market share
        national_prod = merged_df[merged_df['Crop'] == crop]['Production Quantity'].sum()
        state_prod = state_crop_data['Production Quantity'].sum()
        market_share = (state_prod / national_prod) * 100 if national_prod > 0 else 0
        
        # Export metrics
        export_ratio = state_crop_data['Export Volume'].mean() / state_crop_data['Production Quantity'].mean() * 100
        
        # Climate and soil data
        rainfall_var = state_crop_data['Annual Rainfall (mm)'].std() / state_crop_data['Annual Rainfall (mm)'].mean() * 100
        soil_ph = state_crop_data['Soil pH Level'].mean()
        organic_matter = state_crop_data['Organic Matter (%)'].mean()
        
        # Price premium
        national_avg_price = national_results[national_results['Crop'] == crop]['Avg_Price'].mean()
        state_avg_price = state_crop_data['Wholesale Price'].mean()
        price_premium = ((state_avg_price - national_avg_price) / national_avg_price) * 100 if national_avg_price > 0 else 0
        
        rationale_points = []

        # Market Position
        if market_share > 20:
            rationale_points.append(f"• **Market Leader:** Accounts for {market_share:.1f}% of India's {crop} production")
        elif market_share > 10:
            rationale_points.append(f"• **Significant Producer:** Contributes {market_share:.1f}% of national {crop} output")
        else:
            rationale_points.append(f"• **Growing Producer:** Holds {market_share:.1f}% share in {crop} market")
        
        # Export Strength
        if export_ratio > 30:
            rationale_points.append(f"• **Export Powerhouse:** {export_ratio:.1f}% of production is exported globally")
        elif export_ratio > 15:
            rationale_points.append(f"• **Growing Exporter:** {export_ratio:.1f}% export ratio shows international demand")
        else:
            rationale_points.append(f"• **Domestic Focus:** Primarily serves local markets (export ratio: {export_ratio:.1f}%)")
        
        # Rainfall stability
        if rainfall_var < 15:
            rationale_points.append(f"• **Climate Stable:** Highly predictable rainfall (variability: {rainfall_var:.1f}%)")
        elif rainfall_var < 30:
            rationale_points.append(f"• **Moderate Climate Risk:** Rainfall varies by {rainfall_var:.1f}% annually")
        else:
            rationale_points.append(f"• **Climate Challenge:** High rainfall variability ({rainfall_var:.1f}%) requires irrigation planning")
        
        # Soil Quality
        if soil_ph >= 6 and soil_ph <= 7 and organic_matter >= 2:
            rationale_points.append(f"• **Premium Soil:** Ideal pH ({soil_ph:.1f}) and high organic content ({organic_matter:.1f}%)")
        elif (5.5 <= soil_ph < 6) or (7 < soil_ph <= 7.5):
            rationale_points.append(f"• **Good Soil:** Moderate pH ({soil_ph:.1f}) and organic matter ({organic_matter:.1f}%)")
        else:
            rationale_points.append(f"• **Soil Improvement Needed:** pH {soil_ph:.1f}, organic matter {organic_matter:.1f}%")
        
        # Price Advantage
        if price_premium > 10:
            rationale_points.append(f"• **Premium Pricing:** Local prices are {price_premium:.1f}% above national average")
        elif price_premium > 0:
            rationale_points.append(f"• **Slight Price Advantage:** State commands a {price_premium:.1f}% price premium")
        elif price_premium < -10:
            rationale_points.append(f"• **Price Disadvantage:** Prices are {abs(price_premium):.1f}% below national average")
        else:
            rationale_points.append(f"• **Competitive Pricing:** Prices align closely with national averages")
        
        # State-specific tags
        if state == 'Punjab' and crop == 'Wheat':
            rationale_points.append("• **Punjab Special:** Benefits from established wheat procurement system")
        elif state == 'Maharashtra' and crop == 'Grapes':
            rationale_points.append("• **Maharashtra Advantage:** World-class grape processing infrastructure")

        return rationale_points
    
    except Exception as e:
        return [f"Error generating rationale: {str(e)}"]

def analyze_all_states():
    all_results = []
    all_forecasts = {}
    available_states = merged_df['State'].unique()
    
    print("\n" + "="*100)
    print("COMPREHENSIVE AGRICULTURAL ANALYSIS ACROSS ALL INDIAN STATES (10-YEAR FORECAST)")
    print("="*100)
    print(f"\nAvailable states: {', '.join(available_states)}\n")
    
    for state_name in available_states:
        state_df = merged_df[merged_df['State'] == state_name]
        state_results, state_forecasts = analyze_state(state_name, state_df)

        if state_results is not None:
            print_state_report(state_name, state_results, state_forecasts)
            all_results.append(state_results)
            all_forecasts[state_name] = state_forecasts

    if not all_results:
        print("\nNo valid data available for any state.")
        return None, None

    # Combine all results and find top 5 nationally
    national_results = pd.concat(all_results)
    national_results['National_Rank'] = national_results['Score'].rank(ascending=False)
    national_results = national_results.sort_values('National_Rank')
    top_5_national = national_results.head(5)

    # === National Recommendations Summary ===
    national_lines = []
    national_lines.append("="*100)
    national_lines.append("TOP 5 AGRICULTURAL INVESTMENT OPPORTUNITIES ACROSS INDIA (10-YEAR FORECAST)")
    national_lines.append("="*100)

    for i, (_, row) in enumerate(top_5_national.iterrows(), 1):
        national_lines.append(f"\n🏆 RECOMMENDATION #{i}: {row['Crop'].upper()} in {row['State'].upper()}")
        national_lines.append(f"📍 Current Production: {row['Current_Production']:,.2f} tonnes")
        national_lines.append(f"🔮 Forecasted Production (10yr avg): {row['Forecasted_Production']:,.2f} tonnes")
        national_lines.append(f"📈 10-Year Growth Rate: {row['Growth_Rate']:.2%}")
        national_lines.append(f"💰 Average Price: ₹{row['Avg_Price']:,.2f}")
        national_lines.append(f"🌱 Soil Score: {row['Soil_Score']}/5")
        national_lines.append(f"🏆 10-Year Investment Score: {row['Score']:,.2f}")
    
    print("\n".join(national_lines))

    # === Investment Rationales ===
    rationale_lines = []
    rationale_lines.append("\n" + "="*100)
    rationale_lines.append("💡 DATA-DRIVEN INVESTMENT RATIONALE")
    rationale_lines.append("="*100)

    for i, (_, row) in enumerate(top_5_national.iterrows(), 1):
        rationale_lines.append(f"\n🔍 Why invest in {row['Crop']} in {row['State']}?")
        rationale = generate_investment_rationale(row['State'], row['Crop'], merged_df, national_results)
        rationale_lines.extend(rationale)

    print("\n".join(rationale_lines))

    # Save national summary and rationale as report
    full_national_report = "\n".join(national_lines + rationale_lines)
    save_report(full_national_report, "national_top_5_report.txt")

    # === Plot Top 5 National Bar Chart ===
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (_, row) in enumerate(top_5_national.iterrows()):
        ax.bar(f"{row['Crop']}\n({row['State']})", row['Score'], color=colors[i], label=f"#{i+1}")
    
    ax.set_title("Top 5 Agricultural Investment Opportunities Across India\n(10-Year Forecast Period)", pad=20)
    ax.set_xlabel("Crop and State", labelpad=10)
    ax.set_ylabel("10-Year Investment Potential Score", labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title="Ranking")
    fig.tight_layout()
    
    save_plot(fig, "national_top_5_bar_chart.png")

    # === Plot 10-Year Forecast Graphs for Top 5 ===
    print("\n" + "="*100)
    print("10-YEAR FORECAST GRAPHS FOR TOP 5 NATIONAL CROPS")
    print("="*100)

    for i, (_, row) in enumerate(top_5_national.iterrows(), 1):
        crop = row['Crop']
        state = row['State']
        if crop in all_forecasts[state]:
            print(f"\n10-Year Forecast for #{i}: {crop} in {state}")
            plot_state_forecast(state, row, all_forecasts[state])
    
    return national_results, all_forecasts


if __name__ == "__main__":
    analyze_all_states()
    print("✅ All reports and plots saved to results/sectoral/agriculture folders.")
