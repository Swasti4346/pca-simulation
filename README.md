# Personal Carbon Allowance (PCA) Simulation Model

This project features an Agent-Based Model (ABM) built in Python to simulate the economic and distributional impacts of a Personal Carbon Allowance (PCA) policy on Irish households. 

It was built to answer the core question: **Is a Personal Carbon Allowance system administratively feasible, economically robust, and superior under uncertainty when compared to a traditional Carbon Tax?**

## Key Features
1. **Empirically Grounded:** The algorithmic agents are calibrated using primary data from a Carbon Diary Study (baseline emissions), live trading experiments (behavioral mechanics), and CSO data (income distributions).
2. **Dynamic Market Clearing:** The model replaces static calculations with an algorithm that discovers the true market-clearing price of carbon by evaluating the Marginal Abatement Cost (MAC) of 10,000 distinct households.
3. **Distributional Analysis:** Mathematical proof of the PCA's progressive wealth-transfer mechanics versus the regressive nature of flat carbon taxes.

## How to Run locally
To run this application locally on your machine:

1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit interactive dashboard:
   ```bash
   streamlit run app.py
   ```

## Included Scenarios 
The simulation models multiple policy "What Ifs":
*   **The Carbon Tax Benchmark:** Proves the regressive nature of flat carbon prices.
*   **Uniform PCA Allocation:** Proves PCA acts as a progressive wealth transfer.
*   **Equity PCA Allocation:** Demonstrates how tuning the initial allowance allocations further reduces national Gini coefficients and energy poverty.
*   **Behavioral Nudges:** Highlights how increased household elasticity (driven by Carbon App feedback loops) collapses the systemic cost of fighting climate change.
*   **Price Bounds & Energy Shocks:** Stress-tests the allowance market under volatility.
