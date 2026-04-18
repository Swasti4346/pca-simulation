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
        """Generates the household agents with realistic distributions for Ireland."""
        # 1. Income Quintiles (1 to 5)
        quintiles = self.rng.integers(1, 6, size=self.num_households)
        
        # 2. Household Size (1 to 6 people) - vaguely correlated with income
        # Base probabilities for sizes 1,2,3,4,5,6
        sizes = self.rng.choice([1, 2, 3, 4, 5, 6], size=self.num_households, p=[0.25, 0.30, 0.15, 0.15, 0.10, 0.05])
        
        # 3. Baseline Emissions (tonnes CO2/year) - linked to income and size
        # Average Irish household is ~11 tonnes. 
        # Base: 3 tonnes + 1.5 * size + 1.5 * quintile + noise
        base_emissions = 3.0 + (1.5 * sizes) + (1.5 * quintiles) + self.rng.normal(0, 1.5, self.num_households)
        base_emissions = np.maximum(base_emissions, 2.0) # Floor at 2 tonnes
        
        # 4. Price Elasticity (0.1 to 0.5)
        # Higher income tends to have lower elasticity (less responsive to price)
        # Lower income has higher elasticity (more responsive)
        base_elasticity = 0.4 - (quintiles * 0.05) + self.rng.uniform(-0.05, 0.05, self.num_households)
        elasticities = np.clip(base_elasticity, 0.1, 0.5)
        
        # 5. Income (rough approximation in EUR for burden calculation)
        # Q1: 20k, Q2: 40k, Q3: 60k, Q4: 90k, Q5: 130k
        income_map = {1: 20000, 2: 40000, 3: 60000, 4: 90000, 5: 130000}
        incomes = np.array([income_map[q] for q in quintiles]) * self.rng.uniform(0.9, 1.1, self.num_households)
        
        self.agents = pd.DataFrame({
            'id': np.arange(self.num_households),
            'quintile': quintiles,
            'hh_size': sizes,
            'income': incomes,
            'baseline_emissions': base_emissions,
            'elasticity': elasticities
        })
        
        # Abatement slope alpha: determines how much they abate per EUR of carbon price
        # abatement = alpha * Price. 
        # We calibrate so at P=100 EUR, abatement is (elasticity * 20%) of baseline.
        # 100 * alpha = E0 * elast * 0.2  => alpha = E0 * elast * 0.002
        self.agents['alpha'] = self.agents['baseline_emissions'] * self.agents['elasticity'] * 0.002
        
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
            # Energy is more expensive, they abate more naturally without carbon price
            # We simulate this by reducing their baseline emissions by 5%
            self.agents['baseline_emissions'] *= 0.95
            
        elif shock_type == 'behavioral_nudge':
            # Weekly feedback increases compliance and elasticity
            self.agents['elasticity'] = np.clip(self.agents['elasticity'] * 1.5, 0.1, 0.8)
            self.agents['alpha'] = self.agents['baseline_emissions'] * self.agents['elasticity'] * 0.002

    def simulate_market(self, price_floor=None, price_ceiling=None, carbon_tax=None):
        """
        Finds the market clearing price where Total Abatement = Baseline - Cap.
        Or applies a fixed carbon tax.
        Returns the clearing price.
        """
        if carbon_tax is not None:
            # Fixed price, no cap constraint
            P = carbon_tax
        else:
            # Market clearing: sum(alpha_i * P) = sum(E0_i) - Cap
            # P = (sum(E0_i) - Cap) / sum(alpha_i)
            required_abatement = self.agents['baseline_emissions'].sum() - self.cap
            if required_abatement <= 0:
                P = 0
            else:
                total_alpha = self.agents['alpha'].sum()
                P = required_abatement / total_alpha
                
            # Apply bounds
            if price_floor is not None:
                P = max(P, price_floor)
            if price_ceiling is not None:
                P = min(P, price_ceiling)
                
        self.market_price = P
        
        # Calculate final emissions based on price
        # Abatement cannot exceed baseline 
        self.agents['abatement'] = np.clip(self.agents['alpha'] * P, 0, self.agents['baseline_emissions'] * 0.9)
        self.agents['final_emissions'] = self.agents['baseline_emissions'] - self.agents['abatement']
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
