from pca_model import PCASimulation
from analysis import analyze_results, print_summary

def run_scenarios(num_households=5000, target_reduction=0.10):
    results = []
    
    # 1. Baseline PCA, uniform allocation
    print("Running Scenario 1: Baseline Uniform PCA...")
    m1 = PCASimulation(num_households=num_households, random_seed=42)
    m1.generate_households()
    m1.allocate_allowances(method='uniform', cap_reduction=target_reduction)
    m1.simulate_market()
    res1 = analyze_results(m1, "Scenario 1: Uniform Allocation")
    results.append(res1)
    
    # 2. PCA with equity-adjusted allocations
    print("Running Scenario 2: Equity-adjusted PCA...")
    m2 = PCASimulation(num_households=num_households, random_seed=42)
    m2.generate_households()
    m2.allocate_allowances(method='equity', cap_reduction=target_reduction)
    m2.simulate_market()
    res2 = analyze_results(m2, "Scenario 2: Equity Allocation")
    results.append(res2)
    
    # 3. PCA with price ceiling/floor (Cap EUR 200, Floor EUR 20)
    print("Running Scenario 3: Price Bounds PCA...")
    m3 = PCASimulation(num_households=num_households, random_seed=42)
    m3.generate_households()
    m3.allocate_allowances(method='uniform', cap_reduction=target_reduction)
    m3.simulate_market(price_floor=20, price_ceiling=200)
    res3 = analyze_results(m3, "Scenario 3: Price Bounds")
    results.append(res3)
    
    # 4. Energy Price Shock (+20% implied by base emissions dropping naturally)
    print("Running Scenario 4: Energy Price Shock PCA...")
    m4 = PCASimulation(num_households=num_households, random_seed=42)
    m4.generate_households()
    # Apply shock before market trading
    m4.apply_shock(shock_type='energy_price_surge')
    m4.allocate_allowances(method='uniform', cap_reduction=target_reduction)
    m4.simulate_market()
    res4 = analyze_results(m4, "Scenario 4: Energy Shock")
    results.append(res4)
    
    # 5. Flat Carbon Tax (using price from Scenario 1 to compare)
    print("Running Scenario 5: Flat Carbon Tax...")
    m5 = PCASimulation(num_households=num_households, random_seed=42)
    m5.generate_households()
    equiv_tax = res1['Clearing_Price_EUR']
    m5.allocate_allowances(method='uniform', cap_reduction=target_reduction) # just sets cap target for metric but unused
    m5.simulate_market(carbon_tax=equiv_tax)
    res5 = analyze_results(m5, "Scenario 5: Flat Carbon Tax")
    results.append(res5)
    
    # 6. Behavioral Intervention
    print("Running Scenario 6: Behavioral Intervention PCA...")
    m6 = PCASimulation(num_households=num_households, random_seed=42)
    m6.generate_households()
    m6.apply_shock(shock_type='behavioral_nudge')
    m6.allocate_allowances(method='uniform', cap_reduction=target_reduction)
    m6.simulate_market()
    res6 = analyze_results(m6, "Scenario 6: Behavioral Intervention")
    results.append(res6)
    
    return results

