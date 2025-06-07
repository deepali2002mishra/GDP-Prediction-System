import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# Load and clean data
it_df = pd.read_csv(r"D:\Projects\GDP\data\raw\IT_Sector_India_2010_2020.csv")
it_df.replace([np.inf, -np.inf], np.nan, inplace=True)
it_df.dropna(inplace=True)

def analyze_state(state_name, state_df, forecast_years=10):
    try:
        state_df['Revenue_Growth'] = state_df['State_IT_Revenue(Cr)'].pct_change().fillna(0)
        
        model = SARIMAX(
            state_df['State_IT_Revenue(Cr)'],
            exog=state_df[['Repo_Rate(%)', 'Global_Economic_Index']],
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0)
        )
        model_fit = model.fit(disp=False)
        
        forecast = model_fit.get_forecast(
            steps=forecast_years,
            exog=pd.DataFrame({
                'Repo_Rate(%)': [state_df['Repo_Rate(%)'].iloc[-1]] * forecast_years,
                'Global_Economic_Index': [state_df['Global_Economic_Index'].iloc[-1]] * forecast_years
            })
        )
        
        return {
            'state': state_name,
            'current_revenue': state_df['State_IT_Revenue(Cr)'].iloc[-1],
            'avg_growth': state_df['Revenue_Growth'].mean(),
            'unemployment': state_df['Urban_Unemployment_Rate(%)'].iloc[-1],
            'internet_penetration': state_df['Internet_Penetration(%)'].iloc[-1],
            'historical_revenue': state_df['State_IT_Revenue(Cr)'].values,
            'forecast_revenue': forecast.predicted_mean.values,
            'conf_int': forecast.conf_int(),
            'historical_years': state_df['Year'].values,
            'forecast_years': [state_df['Year'].iloc[-1] + i + 1 for i in range(forecast_years)]
        }
    except Exception as e:
        print(f"Error processing {state_name}: {str(e)}")
        return None

def print_state_details(results):
    print("\n" + "="*80)
    print("DETAILED STATE-WISE IT SECTOR PERFORMANCE")
    print("="*80)
    
    for state in results:
        if state:
            growth = (state['forecast_revenue'].mean() - state['current_revenue']) / state['current_revenue']
            print(f"\n📌 {state['state'].upper()}")
            print(f"• Current Revenue: ₹{state['current_revenue']:,.2f} Cr")
            print(f"• Avg Historical Growth: {state['avg_growth']:.2%}")
            print(f"• Urban Unemployment: {state['unemployment']:.2f}%")
            print(f"• Internet Penetration: {state['internet_penetration']:.2f}%")
            print(f"• 10-Year Projected Growth: {growth:.2%}")

def plot_combined_line_graph(results):
    plt.figure(figsize=(14, 7))
    for state in results:
        if state:
            all_years = list(state['historical_years']) + state['forecast_years']
            all_revenue = list(state['historical_revenue']) + list(state['forecast_revenue'])
            
            plt.plot(state['historical_years'], 
                     state['historical_revenue'], 
                     label=f"{state['state']} (Historical)")
            plt.plot(state['forecast_years'], 
                     state['forecast_revenue'], 
                     linestyle='--', 
                     label=f"{state['state']} (Forecast)")
    
    plt.title("IT Revenue Trends Across All States (10-Year Forecast)", pad=20)
    plt.xlabel("Year")
    plt.ylabel("Revenue (₹ Cr)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(r"D:\Projects\GDP\results\sectoral\IT\plots\combined_forecast_trends.png")
    plt.close()

def plot_top3_bar_chart(top_states):
    plt.figure(figsize=(10, 6))
    
    states = [s['state'] for s in top_states if s]
    growth_rates = [
        (s['forecast_revenue'].mean() - s['current_revenue']) / s['current_revenue']
        for s in top_states if s
    ]
    
    if not states:
        print("No valid states for bar chart")
        return
    
    bars = plt.bar(states, growth_rates, color=['gold', 'silver', 'lightblue'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2%}',
                 ha='center', va='bottom')
    
    plt.title("Top 3 States by Projected Growth Rate", pad=20)
    plt.ylabel("10-Year Growth Projection")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(r"D:\Projects\GDP\results\sectoral\IT\plots\top3_growth_bar_chart.png")
    plt.close()

def generate_investment_strategy(top_states):
    output_path = r"D:\Projects\GDP\results\sectoral\IT\reports\top3_investment_strategy.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\nSTRATEGIC INVESTMENT PLAN FOR TOP 3 STATES\n" + "="*80 + "\n")
        
        all_states = [s for s in top_states if s]
        national_unemployment = np.mean([s['unemployment'] for s in all_states])
        national_internet_penetration = np.mean([s['internet_penetration'] for s in all_states])
        national_growth = np.mean([
            (s['forecast_revenue'].mean() - s['current_revenue']) / s['current_revenue']
            for s in all_states
        ])
        
        for i, state in enumerate(all_states[:3], 1):
            growth_rate = (
                (state['forecast_revenue'].mean() - state['current_revenue']) / state['current_revenue']
            )
            revenue_volatility = np.std(state['historical_revenue']) / np.mean(state['historical_revenue'])
            conf_int = state['conf_int']
            conf_width = (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).mean()

            f.write(f"\n🏆 #{i}: {state['state'].upper()} (Projected Growth: {growth_rate:.2%})\n")
            f.write("\nWHY INVEST HERE?\n")
            f.write(f"• Growth rate ({growth_rate:.2%}) {'exceeds' if growth_rate > national_growth else 'is below'} the national average ({national_growth:.2%})\n")
            f.write(f"• Urban Unemployment Rate: {state['unemployment']:.2f}% (National Avg: {national_unemployment:.2f}%)\n")
            f.write(f"• Internet Penetration: {state['internet_penetration']:.2f}% (National Avg: {national_internet_penetration:.2f}%)\n")
            f.write(f"• Revenue Volatility: {revenue_volatility:.2%}\n")
            f.write(f"• Forecast Confidence Interval Width: ₹±{conf_width:.2f} Cr\n")

            f.write("\nCORE ALLOCATION:\n")
            f.write(f"• 50% to established IT firms\n")
            f.write(f"• 30% to digital infrastructure\n")
            f.write(f"• 20% to workforce development\n")
            
            f.write("\nSPECIAL INITIATIVES:\n")
            if growth_rate > 0.2:
                f.write("- Create special economic zone for tech companies\n- Offer 5-year tax holiday for new IT investments\n")
            elif growth_rate > 0.1:
                f.write("- Upgrade existing IT parks with 5G infrastructure\n- Subsidize tech education programs\n")
            else:
                f.write("- Implement business retention grants\n- Develop regional innovation hubs\n")

if __name__ == "__main__":
    print("INDIAN IT SECTOR INVESTMENT ANALYSIS")
    print("="*80)

    all_results = []
    for state_name in it_df['State'].unique():
        state_data = it_df[it_df['State'] == state_name].sort_values('Year')
        result = analyze_state(state_name, state_data)
        all_results.append(result)

    print_state_details(all_results)

    valid_results = [r for r in all_results if r is not None]
    valid_results.sort(key=lambda x: ((x['forecast_revenue'].mean() - x['current_revenue']) / x['current_revenue']), reverse=True)
    top3 = valid_results[:3]

    plot_combined_line_graph(valid_results)
    plot_top3_bar_chart(top3)
    generate_investment_strategy(top3)

    print("\nAnalysis completed successfully!")
