import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pca_model import PCASimulation
from analysis import analyze_results

st.set_page_config(page_title="PCA Simulation Dashboard", layout="wide")

st.title("Personal Carbon Allowance (PCA) Interactive Simulation")
st.markdown("Test the economic and distributional impacts of a Personal Carbon Allowance system for Irish Households.")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Simulation Parameters")

num_households = st.sidebar.slider("Number of Households", min_value=1000, max_value=20000, value=5000, step=1000)
cap_reduction = st.sidebar.slider("National Emission Reduction Cap (%)", min_value=1, max_value=30, value=10, step=1) / 100.0

st.sidebar.subheader("Policy Choices")
allocation_method = st.sidebar.selectbox("Allowance Allocation Method", ["Uniform (Per Capita)", "Equity (Pro-Poor)"])
allocation_key = 'uniform' if 'Uniform' in allocation_method else 'equity'

scenario_type = st.sidebar.radio("Scenario Type", ["Baseline (pure free market)", "Hard Price Bounds", "Energy Price Shock", "Carbon Tax Equivalent", "Behavioral Nudge"])

price_floor = None
price_ceiling = None
carbon_tax = None 
shock = None

if scenario_type == "Hard Price Bounds":
    price_ceiling = st.sidebar.number_input("Price Ceiling (€/tCO2)", min_value=50, max_value=500, value=200)
    price_floor = st.sidebar.number_input("Price Floor (€/tCO2)", min_value=0, max_value=100, value=20)
elif scenario_type == "Energy Price Shock":
    shock = 'energy_price_surge'
elif scenario_type == "Carbon Tax Equivalent":
    carbon_tax = st.sidebar.number_input("Flat Carbon Tax (€/tCO2)", min_value=10, max_value=500, value=210)
elif scenario_type == "Behavioral Nudge":
    shock = 'behavioral_nudge'


if st.sidebar.button("Run Simulation", type="primary"):
    with st.spinner("Simulating..."):
        # Initialize
        model = PCASimulation(num_households=num_households, random_seed=None)
        model.generate_households()
        
        # Apply shocks
        if shock:
            model.apply_shock(shock_type=shock)
            
        # Allocate & Simulate
        model.allocate_allowances(method=allocation_key, cap_reduction=cap_reduction)
        clearing_price = model.simulate_market(price_floor=price_floor, price_ceiling=price_ceiling, carbon_tax=carbon_tax)
        
        # Analyze
        results = analyze_results(model, "Interactive Run")
        df = model.agents
        
        # ----------------- DISPLAY METRICS -----------------
        st.header("Results Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Market Clearing Price", f"€{results['Clearing_Price_EUR']:.2f}")
        col2.metric("CO₂ Reduction Achieved", f"{results['CO2_Reduction_Pct']:.1f}%", f"Target: {cap_reduction*100:.1f}%")
        col3.metric("Gini Inequality Index Impact", f"{results['Gini_Post']:.4f}", f"{results['Gini_Change']:.4f}", delta_color="inverse")
        col4.metric("Total Welfare Cost", f"€{results['Total_Welfare_Cost_M_EUR']:.2f}M")
        
        # ----------------- VISUALIZATIONS -----------------
        st.subheader("Distributional Impact by Income Quintile")
        st.markdown("*A negative income burden % means the quintile made a **profit** from selling allowances.*")
        
        # Burden bar chart
        burden_by_q = df.groupby('quintile')['income_burden_pct'].mean().reset_index()
        fig_burden = px.bar(burden_by_q, x='quintile', y='income_burden_pct', 
                            color='income_burden_pct', color_continuous_scale='RdYlGn_r',
                            labels={'quintile': 'Income Quintile (1 = Lowest, 5 = Highest)', 'income_burden_pct': 'Average Cost Burden (% of Income)'})
        st.plotly_chart(fig_burden, use_container_width=True)
        
        
        st.subheader("Market Mechanics")
        colA, colB = st.columns(2)
        
        with colA:
            # Buyers vs Sellers pie chart
            if carbon_tax:
                st.info("Under a Carbon Tax, all households are 'buyers' (they all pay the tax).")
            else:
                trade_status = df['net_allowances'].apply(lambda x: 'Seller (Emission < Allowance)' if x > 0 else 'Buyer (Emission > Allowance)')
                fig_pie = px.pie(names=trade_status.value_counts().index, values=trade_status.value_counts().values, title="Household Trading Status")
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with colB:
            # Violin plot of emissions pre vs post
            df_melt = pd.melt(df, id_vars=['id', 'quintile'], value_vars=['baseline_emissions', 'final_emissions'], 
                              var_name='Stage', value_name='Emissions (tCO2)')
            df_melt['Stage'] = df_melt['Stage'].replace({'baseline_emissions': 'Pre-Policy (Baseline)', 'final_emissions': 'Post-Policy (Final)'})
            fig_box = px.box(df_melt, x='quintile', y='Emissions (tCO2)', color='Stage', 
                             title="Emissions Spread by Income Quintile",
                             labels={'quintile': 'Income Quintile'})
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Data export
        st.subheader("Agent Data")
        with st.expander("View Raw Household Data"):
            st.dataframe(df.head(100))
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Agent CSV",
                data=csv,
                file_name='agent_simulation_results.csv',
                mime='text/csv',
            )

else:
    st.info("👈 Adjust the parameters on the sidebar and click **Run Simulation** to see the interactive results!")

