"""
Policy Optimization Script with Distributional Equality

This script maximizes GDP growth subject to:
- Revenue neutrality constraint (no increase in deficit)
- Distributional equality constraint (all income groups within 1% of each other)
- National Security (NS) constraints: Mutual exclusivity within NS policy groups

The distributional equality constraint ensures that policy impacts are
distributed fairly across all income groups, with after-tax income changes
for P20, P40-60, P80-100, and P99 differing by no more than 1 percentage point.

Key Features:
1. Distributional Equality: All income groups within 1% of each other
2. Revenue Neutrality: Ensures total dynamic revenue is non-negative
3. National Security (NS) Constraints:
   - NS policy groups (e.g., NS1A, NS1B, NS1C): Only one option per group can be selected
   - Prevents selecting conflicting national security policies
4. Two-stage optimization with tiebreaking on lower-income benefit
"""

from typing import Tuple
import pandas as pd
from gurobipy import Model, GRB, quicksum

from config import COLUMNS, SUPPRESS_GUROBI_OUTPUT, DISTRIBUTIONAL_TOLERANCE
from utils import load_policy_data, display_results_with_distribution


def optimize_policy_selection(
    df_clean: pd.DataFrame,
    ns_groups: dict,
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, float]:
    """
    Two-stage optimization with distributional equality and NS mutual exclusivity constraints.
    
    Stage 1: Maximize GDP subject to revenue neutrality, distributional equality, and NS constraints
        - Finds the maximum achievable GDP growth
        - Ensures total dynamic revenue is non-negative
        - Enforces that all income groups have similar after-tax income changes
        - Enforces NS mutual exclusivity: at most one policy per NS group
        
    Stage 2: Maximize lower-income benefit while maintaining optimal GDP and equality
        - Among solutions achieving optimal GDP from Stage 1
        - Prioritizes policies that benefit lower-income groups
        - Uses lexicographic weighting to prefer P20 > P40-60 > P80-100 > P99
    
    National Security (NS) Mutual Exclusivity:
        For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
        at most one option can be selected. This prevents selecting
        conflicting national security policies within the same category.
    
    Args:
        df_clean: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        verbose: If True, prints progress messages
        
    Returns:
        tuple: (selected_df, gdp_impact, revenue_impact)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - revenue_impact: Total revenue impact achieved
    """
    # Extract data arrays from DataFrame for optimization
    n = len(df_clean)
    gdp = df_clean[COLUMNS["gdp"]].values
    revenue = df_clean[COLUMNS["dynamic_revenue"]].values
    
    # Income percentile impact arrays
    # P20 = bottom 20%, P40-60 = middle class, P80-100 = top 20%, P99 = top 1%
    p20 = df_clean[COLUMNS["p20"]].fillna(0).values
    p40_60 = df_clean[COLUMNS["p40_60"]].fillna(0).values
    p80_100 = df_clean[COLUMNS["p80_100"]].fillna(0).values
    p99 = df_clean[COLUMNS["p99"]].fillna(0).values
    
    # === Stage 1: Maximize GDP with distributional equality ===
    if verbose:
        print("Running optimization (Stage 1: Maximize GDP with Distributional Equality)...")
    
    stage1_model = Model("Stage1_MaximizeGDP_EqualDistribution")
    if SUPPRESS_GUROBI_OUTPUT:
        stage1_model.setParam('OutputFlag', 0)
    
    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = stage1_model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Revenue neutrality (same as basic model)
    stage1_model.addConstr(
        quicksum(revenue[i] * x[i] for i in range(n)) >= 0,
        name="RevenueNeutrality"
    )
    
    # Calculate total after-tax income change for each percentile group
    p20_total = quicksum(p20[i] * x[i] for i in range(n))
    p40_60_total = quicksum(p40_60[i] * x[i] for i in range(n))
    p80_100_total = quicksum(p80_100[i] * x[i] for i in range(n))
    p99_total = quicksum(p99[i] * x[i] for i in range(n))
    
    # Distributional equality constraints: enforce |group1 - group2| <= DISTRIBUTIONAL_TOLERANCE
    # This ensures all income groups experience similar after-tax income changes
    # (within 1 percentage point of each other)
    distribution_pairs = [
        (p20_total, p40_60_total, "P20_P40"),      # Bottom 20% vs Middle class
        (p20_total, p80_100_total, "P20_P80"),     # Bottom 20% vs Top 20%
        (p20_total, p99_total, "P20_P99"),         # Bottom 20% vs Top 1%
        (p40_60_total, p80_100_total, "P40_P80"),  # Middle class vs Top 20%
        (p40_60_total, p99_total, "P40_P99"),      # Middle class vs Top 1%
        (p80_100_total, p99_total, "P80_P99"),     # Top 20% vs Top 1%
    ]
    
    # For each pair of income groups, enforce the absolute difference constraint
    # We model |lhs - rhs| <= DISTRIBUTIONAL_TOLERANCE using:
    # lhs - rhs <= diff AND rhs - lhs <= diff
    for lhs, rhs, name in distribution_pairs:
        # Create auxiliary variable to represent absolute difference
        diff = stage1_model.addVar(lb=0.0, name=f"abs_diff_{name}")
        # Enforce: lhs - rhs <= diff (handles case when lhs > rhs)
        stage1_model.addConstr(lhs - rhs <= diff, name=f"{name}_pos")
        # Enforce: rhs - lhs <= diff (handles case when rhs > lhs)
        stage1_model.addConstr(rhs - lhs <= diff, name=f"{name}_neg")
        # Limit the absolute difference to DISTRIBUTIONAL_TOLERANCE (1 percentage point)
        stage1_model.addConstr(diff <= DISTRIBUTIONAL_TOLERANCE, name=f"{name}_limit")
    
    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    for group, idxs in ns_groups.items():
        stage1_model.addConstr(
            quicksum(x[i] for i in idxs) <= 1,
            name=f"NS_{group}_mutual_exclusivity"
        )
    
    # Objective: Maximize total GDP impact
    stage1_model.setObjective(
        quicksum(gdp[i] * x[i] for i in range(n)),
        GRB.MAXIMIZE
    )
    stage1_model.optimize()
    
    # Store the optimal GDP value for use in Stage 2
    best_gdp = stage1_model.ObjVal
    
    # === Stage 2: Tiebreaking - prioritize lower-income benefit ===
    # Among all solutions that achieve optimal GDP and satisfy equality constraints,
    # select the one that maximizes benefits for lower-income groups
    if verbose:
        print("Running optimization (Stage 2: Maximize Lower-Income Benefit)...")
    
    stage2_model = Model("Stage2_LowerIncomePrioritization")
    if SUPPRESS_GUROBI_OUTPUT:
        stage2_model.setParam('OutputFlag', 0)
    
    # New decision variables for Stage 2
    x2 = stage2_model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Revenue neutrality (same as Stage 1)
    stage2_model.addConstr(
        quicksum(revenue[i] * x2[i] for i in range(n)) >= 0,
        name="RevenueNeutrality"
    )
    
    # Constraint: Must achieve exactly the optimal GDP from Stage 1
    stage2_model.addConstr(
        quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp,
        name="GDPMatch"
    )
    
    # Calculate distributional totals for Stage 2
    p20_sum = quicksum(p20[i] * x2[i] for i in range(n))
    p40_sum = quicksum(p40_60[i] * x2[i] for i in range(n))
    p80_sum = quicksum(p80_100[i] * x2[i] for i in range(n))
    p99_sum = quicksum(p99[i] * x2[i] for i in range(n))
    
    # Distributional equality constraints (same as Stage 1)
    for (lhs, rhs, name) in [
        (p20_sum, p40_sum, "P20_P40"),
        (p20_sum, p80_sum, "P20_P80"),
        (p20_sum, p99_sum, "P20_P99"),
        (p40_sum, p80_sum, "P40_P80"),
        (p40_sum, p99_sum, "P40_P99"),
        (p80_sum, p99_sum, "P80_P99"),
    ]:
        diff = stage2_model.addVar(lb=0.0, name=f"abs_diff_{name}")
        stage2_model.addConstr(lhs - rhs <= diff, name=f"{name}_pos")
        stage2_model.addConstr(rhs - lhs <= diff, name=f"{name}_neg")
        stage2_model.addConstr(diff <= DISTRIBUTIONAL_TOLERANCE, name=f"{name}_limit")
    
    # NS mutual exclusivity constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        stage2_model.addConstr(
            quicksum(x2[i] for i in idxs) <= 1,
            name=f"NS_{group}_mutual_exclusivity"
        )
    
    # Objective: Prioritize lower-income brackets using lexicographic weighting
    # Weights: P20 gets highest priority (1e6), then P40-60 (1e3), then P80-100 (1e1), then P99 (1)
    # This ensures we maximize P20 benefit first, then P40-60, etc.
    stage2_model.setObjective(
        p20_sum * 1e6 + p40_sum * 1e3 + p80_sum * 1e1 + p99_sum,
        GRB.MAXIMIZE
    )
    stage2_model.optimize()
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
    selected_indices = [i for i in range(n) if x2[i].X > 0.5]
    selected_df = df_clean.iloc[selected_indices].copy()
    
    return selected_df, best_gdp, stage2_model.ObjVal


def main() -> None:
    """Main execution function."""
    # Load and clean data
    df_clean, ns_groups = load_policy_data()
    
    # Run optimization
    result_df, gdp_impact, revenue_impact = optimize_policy_selection(
        df_clean, ns_groups
    )
    
    # Display results
    display_results_with_distribution(result_df, gdp_impact, revenue_impact)
    
    # Save to CSV
    result_df.to_csv("max_gdp_equal_distro.csv", index=False)
    print("✓ Results saved to 'max_gdp_equal_distro.csv'\n")


if __name__ == "__main__":
    main()
