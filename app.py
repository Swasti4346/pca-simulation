import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pca_model import PCASimulation
from analysis import analyze_results

st.set_page_config(page_title="PCA Simulation Dashboard", layout="wide")

st.title("🌍 Personal Carbon Allowance (PCA) Interactive Simulation")
st.markdown(
    "Model the **economic and distributional impacts** of a Personal Carbon Allowance "
    "system for Irish households. Adjust the parameters and run the simulation to explore outcomes."
)

# ─────────────────────────── SIDEBAR ───────────────────────────
st.sidebar.header("Simulation Parameters")

num_households = st.sidebar.slider(
    "Number of Households", min_value=1000, max_value=20000, value=5000, step=1000
)
cap_reduction = (
    st.sidebar.slider(
        "National Emission Reduction Cap (%)", min_value=1, max_value=30, value=10, step=1
    )
    / 100.0
)

st.sidebar.subheader("Policy Choices")
allocation_method = st.sidebar.selectbox(
    "Allowance Allocation Method",
    ["Uniform (Per Capita)", "Equity (Pro-Poor)"],
)
allocation_key = "uniform" if "Uniform" in allocation_method else "equity"

scenario_type = st.sidebar.radio(
    "Scenario Type",
    [
        "Baseline (pure free market)",
        "Hard Price Bounds",
        "Energy Price Shock",
        "Carbon Tax Equivalent",
        "Behavioral Nudge",
    ],
)

price_floor = None
price_ceiling = None
carbon_tax = None
shock = None

if scenario_type == "Hard Price Bounds":
    price_ceiling = st.sidebar.number_input(
        "Price Ceiling (€/tCO₂)", min_value=50, max_value=500, value=200
    )
    price_floor = st.sidebar.number_input(
        "Price Floor (€/tCO₂)", min_value=0, max_value=100, value=20
    )
elif scenario_type == "Energy Price Shock":
    shock = "energy_price_surge"
elif scenario_type == "Carbon Tax Equivalent":
    carbon_tax = st.sidebar.number_input(
        "Flat Carbon Tax (€/tCO₂)", min_value=10, max_value=500, value=210
    )
elif scenario_type == "Behavioral Nudge":
    shock = "behavioral_nudge"

# ─────────────────────────── RUN BUTTON ───────────────────────────
if st.sidebar.button("▶ Run Simulation", type="primary"):
    with st.spinner("Simulating market…"):

        model = PCASimulation(num_households=num_households, random_seed=None)
        model.generate_households()

        if shock:
            model.apply_shock(shock_type=shock)

        model.allocate_allowances(method=allocation_key, cap_reduction=cap_reduction)
        clearing_price = model.simulate_market(
            price_floor=price_floor, price_ceiling=price_ceiling, carbon_tax=carbon_tax
        )

        results = analyze_results(model, "Interactive Run")
        df = model.agents

        # ── Quintile labels (FIX: always categorical strings for all charts) ──
        QUINTILE_LABELS = {1: "Q1 (Lowest)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 (Highest)"}
        QUINTILE_ORDER  = list(QUINTILE_LABELS.values())
        df["quintile_label"] = df["quintile"].map(QUINTILE_LABELS)

        # ─────────────────── KEY METRICS ───────────────────
        st.header("📊 Results Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Market Clearing Price", f"€{results['Clearing_Price_EUR']:.2f}/tCO₂")
        col2.metric(
            "CO₂ Reduction Achieved",
            f"{results['CO2_Reduction_Pct']:.1f}%",
            f"Target: {cap_reduction * 100:.1f}%",
        )
        col3.metric(
            "Gini Index Change",
            f"{results['Gini_Post']:.4f}",
            f"{results['Gini_Change']:+.4f}",
            delta_color="inverse",
        )
        col4.metric(
            "Total Economy-Wide Welfare Cost",
            f"€{results['Total_Welfare_Cost_M_EUR']:.2f}M",
        )

        st.markdown("---")

        # ═══════════════════════════════════════════════════════════
        # 1.  DISTRIBUTIONAL IMPACT (FIXED: string categories, symmetric colour scale)
        # ═══════════════════════════════════════════════════════════
        st.subheader("📉 Distributional Impact by Income Quintile")

        st.info(
            "**Why does Q1 (Lowest Income) show a negative / profit?** \n\n"
            "Under PCA, every person receives the **same per-capita allowance** regardless of income. "
            "But low-income households own fewer cars, live in smaller homes, and fly less — so they "
            "**emit far less CO₂ than their allocation entitles them to**. "
            "They can therefore **sell their surplus allowances** on the carbon market and receive cash. \n\n"
            "This is the core progressive feature of PCA: it acts as a **wealth transfer from high to low income households**, "
            "unlike a flat carbon tax which is regressive (hits the poor hardest)."
        )

        burden_by_q = (
            df.groupby("quintile_label")["income_burden_pct"]
            .mean()
            .reindex(QUINTILE_ORDER)
            .reset_index()
        )
        burden_by_q.columns = ["Quintile", "Avg Net Cost Burden (% of Income)"]

        max_abs = max(abs(burden_by_q["Avg Net Cost Burden (% of Income)"]).max(), 0.01)

        fig_burden = px.bar(
            burden_by_q,
            x="Quintile",
            y="Avg Net Cost Burden (% of Income)",
            color="Avg Net Cost Burden (% of Income)",
            color_continuous_scale="RdYlGn_r",
            color_continuous_midpoint=0,
            range_color=[-max_abs, max_abs],
            category_orders={"Quintile": QUINTILE_ORDER},
            text="Avg Net Cost Burden (% of Income)",
        )
        fig_burden.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_burden.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)

        # Annotate the two sides of the zero line
        fig_burden.add_annotation(
            x=0.02, y=1.08, xref="paper", yref="paper",
            text="⬇ Below zero = NET SELLER (profits from selling surplus allowances)",
            showarrow=False, font=dict(size=11, color="#34d399"), align="left"
        )
        fig_burden.add_annotation(
            x=0.02, y=1.02, xref="paper", yref="paper",
            text="⬆ Above zero = NET BUYER (must purchase extra allowances / pay cost)",
            showarrow=False, font=dict(size=11, color="#f87171"), align="left"
        )
        fig_burden.update_layout(
            xaxis_title="Income Quintile — ordered lowest to highest income",
            yaxis_title="Average Net Cost Burden (% of Household Income)",
            coloraxis_showscale=True,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=80),
        )
        st.plotly_chart(fig_burden, use_container_width=True)

        # ═══════════════════════════════════════════════════════════
        # 2.  NET MONETARY TRANSFER PER QUINTILE (€ absolute terms)
        # ═══════════════════════════════════════════════════════════
        st.subheader("💶 Average Net Monetary Transfer per Household by Quintile")
        st.markdown(
            "_Shows the **absolute euro amount** gained (profit) or lost (cost) per household. "
            "This is the financial impact AFTER both abatement costs and allowance trading revenue._"
        )

        net_transfer = (
            df.groupby("quintile_label")["total_policy_cost"]
            .mean()
            .reindex(QUINTILE_ORDER)
            .reset_index()
        )
        net_transfer.columns = ["Quintile", "Avg Net Cost (€/household)"]

        max_abs_t = max(abs(net_transfer["Avg Net Cost (€/household)"]).max(), 1)
        fig_transfer = px.bar(
            net_transfer,
            x="Quintile",
            y="Avg Net Cost (€/household)",
            color="Avg Net Cost (€/household)",
            color_continuous_scale="RdYlGn_r",
            color_continuous_midpoint=0,
            range_color=[-max_abs_t, max_abs_t],
            category_orders={"Quintile": QUINTILE_ORDER},
            text="Avg Net Cost (€/household)",
        )
        fig_transfer.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
        fig_transfer.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
        fig_transfer.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_transfer, use_container_width=True)

        # ═══════════════════════════════════════════════════════════
        # 3.  MARKET MECHANICS
        # ═══════════════════════════════════════════════════════════
        st.subheader("⚖️ Market Mechanics")
        colA, colB = st.columns(2)

        with colA:
            if carbon_tax:
                st.info(
                    "**Carbon Tax:** All households pay a flat rate per tonne of emissions. "
                    "There is no trading — every household is a net buyer."
                )
            else:
                trade_counts = (
                    df["net_allowances"]
                    .apply(
                        lambda x: "🟢 Seller (Surplus Allowances)"
                        if x > 0
                        else "🔴 Buyer (Deficit — must purchase)"
                    )
                    .value_counts()
                )
                fig_pie = px.pie(
                    names=trade_counts.index,
                    values=trade_counts.values,
                    title="Household Trading Status",
                    color_discrete_map={
                        "🟢 Seller (Surplus Allowances)": "#10b981",
                        "🔴 Buyer (Deficit — must purchase)": "#ef4444",
                    },
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

        with colB:
            # Market Supply/Demand — Aggregate MAC Curve
            prices = np.linspace(0, max(clearing_price * 2, 50), 200)
            total_demand = [
                (df["baseline_emissions"] - np.clip(df["alpha"] * p, 0, df["baseline_emissions"] * 0.9)).sum()
                for p in prices
            ]
            supply_level = model.cap

            fig_mac = go.Figure()
            fig_mac.add_trace(
                go.Scatter(
                    x=total_demand,
                    y=prices,
                    mode="lines",
                    name="Aggregate Demand for Allowances",
                    line=dict(color="#60a5fa", width=2),
                )
            )
            fig_mac.add_vline(
                x=supply_level,
                line_dash="dash",
                line_color="#10b981",
                annotation_text=f"Cap Supply = {supply_level:,.0f} tCO₂",
                annotation_position="top right",
            )
            if not carbon_tax:
                fig_mac.add_hline(
                    y=clearing_price,
                    line_dash="dot",
                    line_color="#f59e0b",
                    annotation_text=f"Clearing Price = €{clearing_price:.1f}",
                    annotation_position="right",
                )
                fig_mac.add_trace(
                    go.Scatter(
                        x=[supply_level],
                        y=[clearing_price],
                        mode="markers",
                        marker=dict(color="#f59e0b", size=12, symbol="circle"),
                        name="Market Equilibrium",
                    )
                )
            fig_mac.update_layout(
                title="Carbon Market: Demand vs Fixed Supply",
                xaxis_title="Total Emissions / Allowances Demanded (tCO₂)",
                yaxis_title="Carbon Price (€/tCO₂)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_mac, use_container_width=True)

        # ═══════════════════════════════════════════════════════════
        # 4.  EMISSIONS SPREAD (FIXED: string categories)
        # ═══════════════════════════════════════════════════════════
        st.subheader("🌱 Emissions Distribution: Pre vs Post Policy")
        df_melt = pd.melt(
            df,
            id_vars=["id", "quintile_label"],
            value_vars=["baseline_emissions", "final_emissions"],
            var_name="Stage",
            value_name="Emissions (tCO₂/year)",
        )
        df_melt["Stage"] = df_melt["Stage"].replace(
            {
                "baseline_emissions": "📊 Pre-Policy (Baseline)",
                "final_emissions": "✅ Post-Policy (Final)",
            }
        )
        fig_box = px.box(
            df_melt,
            x="quintile_label",
            y="Emissions (tCO₂/year)",
            color="Stage",
            category_orders={
                "quintile_label": QUINTILE_ORDER,
                "Stage": ["📊 Pre-Policy (Baseline)", "✅ Post-Policy (Final)"],
            },
            title="Emissions Spread by Income Quintile",
            labels={"quintile_label": "Income Quintile"},
            color_discrete_map={
                "📊 Pre-Policy (Baseline)": "#f87171",
                "✅ Post-Policy (Final)": "#34d399",
            },
        )
        fig_box.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

        # ═══════════════════════════════════════════════════════════
        # 5.  GINI INEQUALITY BEFORE vs AFTER
        # ═══════════════════════════════════════════════════════════
        st.subheader("⚖️ Inequality Impact (Gini Coefficient)")

        gini_df = pd.DataFrame(
            {
                "Stage": ["Pre-Policy", "Post-Policy"],
                "Gini Coefficient": [results["Gini_Pre"], results["Gini_Post"]],
            }
        )
        direction = "reduced" if results["Gini_Change"] < 0 else "increased"
        st.markdown(
            f"The Gini coefficient **{direction}** by `{abs(results['Gini_Change']):.4f}` "
            f"({'more equal' if direction == 'reduced' else 'less equal'} income distribution after policy)."
        )
        fig_gini = px.bar(
            gini_df,
            x="Stage",
            y="Gini Coefficient",
            color="Stage",
            color_discrete_map={"Pre-Policy": "#94a3b8", "Post-Policy": "#2dd4bf"},
            text="Gini Coefficient",
            range_y=[0, max(results["Gini_Pre"], results["Gini_Post"]) * 1.2],
        )
        fig_gini.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_gini.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gini, use_container_width=True)

        # ═══════════════════════════════════════════════════════════
        # 6.  DATA EXPORT
        # ═══════════════════════════════════════════════════════════
        st.subheader("🗂️ Agent Data")
        with st.expander("View Raw Household Data (first 100 rows)"):
            display_cols = [
                "id", "quintile", "hh_size", "income",
                "baseline_emissions", "final_emissions", "abatement",
                "allocation", "net_allowances", "financial_impact",
                "total_policy_cost", "income_burden_pct",
            ]
            st.dataframe(df[display_cols].head(100))
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Agent CSV",
                data=csv,
                file_name="pca_agent_results.csv",
                mime="text/csv",
            )

else:
    st.info(
        "👈 Adjust the parameters on the sidebar and click **▶ Run Simulation** "
        "to see the interactive results."
    )
    st.markdown(
        """
        ### How the Simulation Works
        | Component | Description |
        |---|---|
        | **Households** | Synthetic Irish households with income, size, emissions & price elasticity |
        | **Allowances** | Each household receives a CO₂ allowance based on the chosen allocation rule |
        | **Market** | A carbon permit market clears at the price where total abatement = emissions gap |
        | **Burden** | Net cost = abatement costs − trading revenue, expressed as % of income |

        **Key economic insight:** Under a uniform per-capita PCA, lower-income households (who emit less)
        tend to receive more allowances than they need and *profit* from selling the surplus.
        Higher-income households (who emit more) must *buy* additional permits, resulting in a positive
        cost burden. This gives PCA a natural **progressive** (pro-poor) character compared to a flat
        carbon tax.
        """
    )
