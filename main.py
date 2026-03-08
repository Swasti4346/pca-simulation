import pandas as pd
from scenarios import run_scenarios

def main():
    print("Starting Personal Carbon Allowance (PCA) Simulation")
    print("---------------------------------------------------")
    
    # Run the simulation for 1000 households (scaled)
    results = run_scenarios(num_households=10000, target_reduction=0.10)
    
    df_results = pd.DataFrame(results)
    
    # Format and display
    print("\n--- Summary of Results ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    # Format floats for cleaner printing
    format_mapping = {
        'CO2_Reduction_Pct': '{:.1f}%',
        'Clearing_Price_EUR': '€{:.2f}',
        'Gini_Pre': '{:.4f}',
        'Gini_Post': '{:.4f}',
        'Gini_Change': '{:.4f}',
        'Total_Welfare_Cost_M_EUR': '€{:.2f}M',
        'Net_Buyers_Pct': '{:.1f}%',
        'Net_Sellers_Pct': '{:.1f}%',
        'Burden_Q1_Pct': '{:.2f}%',
        'Burden_Q5_Pct': '{:.2f}%'
    }
    
    formatted_df = df_results.copy()
    for col, format_str in format_mapping.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: format_str.format(x))
            
    print(formatted_df.to_string(index=False))
    
    # Save results to CSV for external usage
    formatted_df.to_csv("simulation_results.csv", index=False)
    print("\nResults saved to 'simulation_results.csv'")

if __name__ == "__main__":
    main()
