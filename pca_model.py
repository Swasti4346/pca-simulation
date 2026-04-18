import numpy as np
import pandas as pd

class PCASimulation:
    def __init__(self, num_households=5000, random_seed=42):
        self.num_households = num_households
        self.rng = np.random.default_rng(random_seed)
        self.agents = None
        self.market_price = 0
        self.total_emissions = 0
        self.cap = 0
        
    def generate_households(self):
        """Generates household agents calibrated to Irish CSO / ESRI data."""
        # 1. Income Quintiles (equal split: 1=lowest, 5=highest)
        quintiles = self.rng.integers(1, 6, size=self.num_households)

        # 2. Household Size — POSITIVELY correlated with income quintile.
        #    In Ireland, larger families tend to have higher household income.
        #    Source: CSO Household Budget Survey 2015-16.
        #    Q1: many elderly singles / single parents → mean ~1.8 persons
        #    Q5: couples with children, multi-earner → mean ~3.2 persons
        size_probs = {
            1: ([1, 2, 3, 4],       [0.50, 0.30, 0.15, 0.05]),
            2: ([1, 2, 3, 4, 5],    [0.30, 0.35, 0.20, 0.10, 0.05]),
            3: ([1, 2, 3, 4, 5],    [0.20, 0.30, 0.25, 0.18, 0.07]),
            4: ([1, 2, 3, 4, 5, 6], [0.10, 0.25, 0.30, 0.22, 0.10, 0.03]),
            5: ([1, 2, 3, 4, 5, 6], [0.05, 0.20, 0.30, 0.27, 0.13, 0.05]),
        }
        sizes = np.zeros(self.num_households, dtype=int)
        for q in range(1, 6):
            mask = quintiles == q
            opts, probs = size_probs[q]
            sizes[mask] = self.rng.choice(opts, size=int(mask.sum()), p=probs)

        # 3. Baseline Emissions (tCO₂/year) — linked to income quintile AND size.
        #    Irish average ≈ 10–12 tCO₂/hh/yr (SEAI 2022).
        #    Q1 avg ≈ 7 t, Q5 avg ≈ 15 t (ESRI / EPA distributional analysis).
        #    Formula: 2.0 + 1.2*size + 1.2*quintile + noise
        base_emissions = (
            2.0 + (1.2 * sizes) + (1.2 * quintiles)
            + self.rng.normal(0, 1.2, self.num_households)
        )
        base_emissions = np.maximum(base_emissions, 1.5)  # floor at 1.5 t

        # 4. Price Elasticity of Demand for Carbon (magnitude).
        #    Lower-income households are MORE price-sensitive (higher elasticity)
        #    as energy expenditure is a higher share of their budget.
        #    Range 0.10–0.45 (consistent with ESRI Irish energy demand estimates).
        base_elasticity = 0.40 - (quintiles * 0.05) + self.rng.uniform(-0.05, 0.05, self.num_households)
        elasticities = np.clip(base_elasticity, 0.10, 0.45)

        # 5. Disposable Household Income (EUR/year).
        #    Calibrated to CSO Household Budget Survey 2019-20 disposable income.
        #    Q1≈€22k, Q2≈€38k, Q3≈€53k, Q4≈€72k, Q5≈€110k
        income_map = {1: 22000, 2: 38000, 3: 53000, 4: 72000, 5: 110000}
        incomes = (
            np.array([income_map[q] for q in quintiles])
            * self.rng.uniform(0.88, 1.12, self.num_households)
        )

        self.agents = pd.DataFrame({
            'id': np.arange(self.num_households),
            'quintile': quintiles,
            'hh_size': sizes,
            'income': incomes,
            'baseline_emissions': base_emissions,
            'elasticity': elasticities,
        })

        # Abatement cost function: abatement_i = alpha_i * P
        # Calibrated so at P = €100/tCO₂, abatement = elasticity * 18% of baseline
        # (conservative, consistent with Irish short-run energy demand literature)
        self.agents['alpha'] = self.agents['baseline_emissions'] * self.agents['elasticity'] * 0.0018

        
    def allocate_allowances(self, method='uniform', cap_reduction=0.10):
        """
        Allocates allowances to households.
        cap_reduction: % reduction from baseline total emissions.
        methods: 'uniform' (per capita), 'equity' (progressive based on income quintile)
        """
        total_baseline = self.agents['baseline_emissions'].sum()
        self.cap = total_baseline * (1.0 - cap_reduction)
        
        if method == 'uniform':
            # Uniform per capita
            total_people = self.agents['hh_size'].sum()
            per_capita_allowance = self.cap / total_people
            self.agents['allocation'] = self.agents['hh_size'] * per_capita_allowance
            
        elif method == 'equity':
            # Equity-adjusted: lower quintiles get more per capita
            # Weight factors: Q1: 1.4, Q2: 1.2, Q3: 1.0, Q4: 0.8, Q5: 0.6
            weight_map = {1: 1.4, 2: 1.2, 3: 1.0, 4: 0.8, 5: 0.6}
            weights = self.agents['quintile'].map(weight_map)
            weighted_people = self.agents['hh_size'] * weights
            per_weighted_capita = self.cap / weighted_people.sum()
            self.agents['allocation'] = weighted_people * per_weighted_capita
            
        elif method == 'flat':
            # Every household gets exactly the same regardless of size
            self.agents['allocation'] = self.cap / self.num_households
            
    def apply_shock(self, shock_type=None):
        """Applies external shocks like energy price increases or behavioral nudges."""
        if shock_type == 'energy_price_surge':
            # A 5% autonomous reduction in baseline emissions (e.g. fuel price surge
            # already prompting behaviour change before any carbon policy).
            # IMPORTANT: alpha must also be recalculated because it is proportional
            # to baseline_emissions — otherwise the abatement curve is mis-calibrated.
            self.agents['baseline_emissions'] *= 0.95
            self.agents['alpha'] = (
                self.agents['baseline_emissions']
                * self.agents['elasticity']
                * 0.0018   # consistent with generate_households calibration
            )

        elif shock_type == 'behavioral_nudge':
            # Weekly carbon-footprint feedback boosts price responsiveness.
            # Elasticity is clipped at 0.80 (hard physical limit — you can't reduce
            # energy use to zero).  Use the SAME alpha constant (0.0018) as
            # generate_households so the abatement function stays consistent.
            self.agents['elasticity'] = np.clip(
                self.agents['elasticity'] * 1.5, 0.10, 0.80
            )
            self.agents['alpha'] = (
                self.agents['baseline_emissions']
                * self.agents['elasticity']
                * 0.0018
            )

    def simulate_market(self, price_floor=None, price_ceiling=None, carbon_tax=None):
        """
        Finds the market-clearing carbon price such that aggregate abatement
        exactly meets the required emissions reduction (Baseline − Cap).

        The aggregate abatement function is linear: A_total(P) = sum(alpha_i) * P
        subject to each household's abatement being capped at 90 % of their
        baseline (technological feasibility limit).  We therefore solve
        iteratively rather than analytically to guarantee clearing accuracy
        when the cap is binding for some agents.

        Returns the clearing price.
        """
        if carbon_tax is not None:
            # Fixed price instrument — no quantity cap is enforced.
            P = float(carbon_tax)
        else:
            required_abatement = (
                self.agents['baseline_emissions'].sum() - self.cap
            )
            if required_abatement <= 0:
                P = 0.0
            else:
                # ── Analytical first estimate (unclipped) ───────────────────
                # A_total = sum(alpha_i) * P  =>  P = required / sum(alpha_i)
                P = required_abatement / self.agents['alpha'].sum()

                # ── Iterative refinement to account for 90 % abatement cap ─
                # When some agents are capped, their marginal abatement no
                # longer responds to price; the uncapped agents must work
                # harder, raising the equilibrium price.
                for _ in range(20):          # converges in < 5 iterations
                    abate = np.clip(
                        self.agents['alpha'] * P,
                        0,
                        self.agents['baseline_emissions'] * 0.9,
                    )
                    gap = required_abatement - abate.sum()
                    if abs(gap) < 1e-3:      # tCO₂ tolerance
                        break
                    # Uncapped agents still have residual alpha response
                    uncapped_mask = (
                        self.agents['alpha'] * P
                        < self.agents['baseline_emissions'] * 0.9
                    )
                    residual_alpha = self.agents.loc[uncapped_mask, 'alpha'].sum()
                    if residual_alpha < 1e-9:
                        break               # everyone is capped — can't do more
                    P += gap / residual_alpha
                    P = max(P, 0.0)

            # Apply regulatory price bounds
            if price_floor is not None:
                P = max(P, float(price_floor))
            if price_ceiling is not None:
                P = min(P, float(price_ceiling))

        self.market_price = P

        # ── Final emissions ──────────────────────────────────────────────────
        # Abatement is capped at 90 % of baseline (technological upper bound)
        self.agents['abatement'] = np.clip(
            self.agents['alpha'] * P,
            0,
            self.agents['baseline_emissions'] * 0.9,
        )
        self.agents['final_emissions'] = (
            self.agents['baseline_emissions'] - self.agents['abatement']
        )
        self.total_emissions = self.agents['final_emissions'].sum()
        
        # Calculate financial outcomes
        if carbon_tax is not None:
            # Under carbon tax, they pay per tonne of final emissions (no trading)
            self.agents['net_allowances'] = -self.agents['final_emissions']  # Always a net buyer
            self.agents['financial_impact'] = self.agents['net_allowances'] * P  # All negative (cost)
            self.agents['allocation'] = 0
            self.agents['buyer'] = True
            self.agents['seller'] = False
        else:
            # Net allowances: Allocation - Emissions. Positive = Seller, Negative = Buyer
            self.agents['net_allowances'] = self.agents['allocation'] - self.agents['final_emissions']
            self.agents['financial_impact'] = self.agents['net_allowances'] * P # + is profit, - is cost
            self.agents['buyer'] = self.agents['net_allowances'] < 0
            self.agents['seller'] = self.agents['net_allowances'] > 0
            
        # Cost burden as % of income
        # If financial_impact is negative, burden is positive. We also add abatement costs.
        # Abatement cost = integral of MAC = 0.5 * abatement^2 / alpha (since A = alpha * P => P = A/alpha)
        # Or more simply, cost area = 0.5 * P * Abatement
        self.agents['abatement_cost'] = 0.5 * P * self.agents['abatement']
        
        # Total cost = Abatement Cost - Financial Impact (since financial impact is + for profit)
        self.agents['total_policy_cost'] = self.agents['abatement_cost'] - self.agents['financial_impact']
        self.agents['income_burden_pct'] = (self.agents['total_policy_cost'] / self.agents['income']) * 100
        
        return self.market_price
