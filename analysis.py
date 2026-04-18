import numpy as np
import pandas as pd

def calculate_gini(incomes):
    """Calculates the Gini coefficient of a numpy array of incomes."""
    incomes = np.sort(np.asarray(incomes))
    n = len(incomes)
    if n == 0 or np.sum(incomes) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n  - 1) * incomes)) / (n * np.sum(incomes))

def analyze_results(model, scenario_name="Baseline"):
    """
    Computes key metrics from the simulation results:
    - Pre/Post Gini
    - Total Welfare Cost
    - Total CO2 Reduction
    - Clearing Price
    - Distributional impact (average burden by bracket)
    """
    df = model.agents
    
    pre_income = df['income'].values
    # Clip to zero: a household cannot have negative post-policy income in this
    # model; very large costs are capped at full income loss for Gini purposes.
    post_income = np.maximum(pre_income - df['total_policy_cost'].values, 0.0)
    
    gini_pre = calculate_gini(pre_income)
    gini_post = calculate_gini(post_income)
    
    total_baseline_emissions = df['baseline_emissions'].sum()
    total_final_emissions = df['final_emissions'].sum()
    emissions_reduction = total_baseline_emissions - total_final_emissions
    reduction_pct = (emissions_reduction / total_baseline_emissions) * 100
    
    # Net buyers / sellers
    net_buyers  = int((df['net_allowances'] < 0).sum()) if 'net_allowances' in df.columns else len(df)
    net_sellers = int((df['net_allowances'] > 0).sum()) if 'net_allowances' in df.columns else 0
    
    # Total welfare cost = sum of all abatement costs net of trading gains.
    # In a perfectly clearing market this equals aggregate abatement cost only
    # (transfers between buyers and sellers cancel out).
    total_welfare_cost = df['total_policy_cost'].sum()
    
    avg_price = model.market_price
    
    # Calculate impact by quintile
    impact_by_quintile = df.groupby('quintile')['income_burden_pct'].mean().to_dict()
    
    metrics = {
        'Scenario': scenario_name,
        'CO2_Reduction_Pct': reduction_pct,
        'Clearing_Price_EUR': avg_price,
        'Gini_Pre': gini_pre,
        'Gini_Post': gini_post,
        'Gini_Change': gini_post - gini_pre,
        'Total_Welfare_Cost_M_EUR': total_welfare_cost / 1_000_000,
        'Net_Buyers_Pct': (net_buyers / len(df)) * 100,
        'Net_Sellers_Pct': (net_sellers / len(df)) * 100,
        'Burden_Q1_Pct': impact_by_quintile.get(1, 0),
        'Burden_Q5_Pct': impact_by_quintile.get(5, 0)
    }
    
    return metrics

def print_summary(metrics):
    print(f"--- {metrics['Scenario']} ---")
    print(f"Clearing Price: €{metrics['Clearing_Price_EUR']:.2f}")
    print(f"Emissions Reduction: {metrics['CO2_Reduction_Pct']:.1f}%")
    print(f"Gini Pre: {metrics['Gini_Pre']:.4f} | Gini Post: {metrics['Gini_Post']:.4f} (Change: {metrics['Gini_Change']:.4f})")
    print(f"Cost Burden: Q1: {metrics['Burden_Q1_Pct']:.2f}% | Q5: {metrics['Burden_Q5_Pct']:.2f}%")
    print(f"Net Buyers: {metrics['Net_Buyers_Pct']:.1f}% | Net Sellers: {metrics['Net_Sellers_Pct']:.1f}%")
    print("")
