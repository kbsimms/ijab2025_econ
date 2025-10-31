"""
Policy Optimization Script with Equity and National Security Constraints

This script maximizes GDP growth subject to:
- Fiscal constraint: Revenue neutrality (no increase in deficit)
- Economic constraints: Non-negative capital stock, jobs, and wage rate
- Equity constraints: Progressive distribution favoring lower/middle income groups
- National security constraints: Minimum spending requirement, mutual exclusivity

Key Features:
1. Equity Constraints: Ensures lower and middle income groups (P20, P40-60) benefit
   at least as much as upper income groups (P80-100, P99)
2. National Security (NS) Constraints:
   - NS1-NS7 groups (A/B/C options): Only one option per group can be selected
   - Configurable minimum spending requirement on NS1-NS7 selections
3. Two-stage optimization to find best GDP with tiebreaking on revenue

Usage:
    python max_gdp_defense.py                    # Default: $3,000B requirement
    python max_gdp_defense.py --spending 3000    # Explicit: $3,000B requirement
    python max_gdp_defense.py --spending 4000    # Increased: $4,000B requirement
"""

import argparse
from typing import Tuple
import pandas as pd
from gurobipy import Model, GRB, quicksum

from config import (
    COLUMNS,
    SUPPRESS_GUROBI_OUTPUT,
    EPSILON,
    DISTRIBUTIONAL_TOLERANCE,
    DEFENSE_SPENDING
)
from utils import load_policy_data, get_ns_strict_indices, display_results


def optimize_policy_selection(
    df: pd.DataFrame,
    ns_groups: dict,
    ns_strict_indices: list,
    min_ns_spending: int = DEFENSE_SPENDING["baseline"],
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, float]:
    """
    Two-stage optimization with equity and national security constraints.
    
    Stage 1: Maximize GDP subject to all constraints
        - Fiscal: Revenue neutrality (sum of dynamic revenue >= 0)
        - Economic: Non-negative capital stock, jobs, wage rate
        - Equity: Progressive distribution (P20, P40-60 >= P80-100, P99)
        - NS mutual exclusivity: At most one policy per NS group
        - NS spending minimum: At least min_ns_spending on NS1-NS7 policies
        
    Stage 2: Maximize revenue while maintaining optimal GDP
        - Same constraints as Stage 1
        - Additionally constrains GDP to equal the optimal value from Stage 1
        - Breaks ties by maximizing revenue surplus
    
    Equity Constraints Explained:
        The equity constraints ensure progressive policy impacts:
        1. P20 and P40-60 must be within 1% of each other (protect both groups equally)
        2. P20 and P40-60 must benefit at least as much as P80-100 and P99
        3. Combined P20+P40-60 must exceed combined P80-100+P99
        
    Args:
        df: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        ns_strict_indices: List of indices for NS1-NS7 policies
        min_ns_spending: Minimum NS spending in billions (default: 3000)
        verbose: If True, prints progress messages
        
    Returns:
        tuple: (selected_df, gdp_impact, revenue_impact)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - revenue_impact: Total revenue impact achieved
    """
    # Extract data arrays from DataFrame for optimization
    n = len(df)
    indices = range(n)
    
    # Convert DataFrame columns to numpy arrays for efficient access
    gdp = df[COLUMNS["gdp"]].values
    revenue = df[COLUMNS["dynamic_revenue"]].values
    capital = df[COLUMNS["capital"]].values
    jobs = df[COLUMNS["jobs"]].values
    wage = df[COLUMNS["wage"]].values
    p20_arr = df[COLUMNS["p20"]].values
    p40_arr = df[COLUMNS["p40_60"]].values
    p80_arr = df[COLUMNS["p80_100"]].values
    p99_arr = df[COLUMNS["p99"]].values
    
    # === Stage 1: Maximize GDP ===
    if verbose:
        print(f"Running optimization (Stage 1: Maximize GDP)...")
        print(f"  NS spending requirement: ${min_ns_spending:,}B")
    
    stage1_model = Model("Stage1_MaximizeGDP")
    if SUPPRESS_GUROBI_OUTPUT:
        stage1_model.setParam("OutputFlag", 0)
    
    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = stage1_model.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize total GDP impact
    stage1_model.setObjective(
        quicksum(x[i] * gdp[i] for i in indices),
        GRB.MAXIMIZE
    )
    
    # === Constraints ===
    
    # Fiscal constraint: Total dynamic revenue must be non-negative
    # Ensures the policy package doesn't increase the deficit
    stage1_model.addConstr(
        quicksum(x[i] * revenue[i] for i in indices) >= 0,
        name="RevenueNeutrality"
    )
    
    # Economic constraints: Ensure positive economic impacts
    # Capital stock change must be non-negative
    stage1_model.addConstr(
        quicksum(x[i] * capital[i] for i in indices) >= 0,
        name="CapitalStock"
    )
    # Job creation must be non-negative
    stage1_model.addConstr(
        quicksum(x[i] * jobs[i] for i in indices) >= 0,
        name="Jobs"
    )
    # Wage rate change must be non-negative
    stage1_model.addConstr(
        quicksum(x[i] * wage[i] for i in indices) >= 0,
        name="WageRate"
    )
    
    # Equity constraints: Ensure progressive distribution
    # Calculate total after-tax income change for each percentile group
    p20 = quicksum(x[i] * p20_arr[i] for i in indices)
    p40 = quicksum(x[i] * p40_arr[i] for i in indices)
    p80 = quicksum(x[i] * p80_arr[i] for i in indices)
    p99 = quicksum(x[i] * p99_arr[i] for i in indices)
    
    # Constraint 1: P20 and P40-60 within 1% of each other (protect both equally)
    stage1_model.addConstr(
        p20 - p40 <= DISTRIBUTIONAL_TOLERANCE,
        name="P20_P40_upper"
    )
    stage1_model.addConstr(
        p40 - p20 <= DISTRIBUTIONAL_TOLERANCE,
        name="P40_P20_upper"
    )
    
    # Constraint 2: Lower/middle income groups must benefit at least as much as upper groups
    # P20 >= P99 (bottom 20% benefits at least as much as top 1%)
    stage1_model.addConstr(p20 - p99 >= EPSILON, name="P20_ge_P99")
    # P40-60 >= P99 (middle class benefits at least as much as top 1%)
    stage1_model.addConstr(p40 - p99 >= EPSILON, name="P40_ge_P99")
    # P20 >= P80-100 (bottom 20% benefits at least as much as top 20%)
    stage1_model.addConstr(p20 - p80 >= EPSILON, name="P20_ge_P80")
    # P40-60 >= P80-100 (middle class benefits at least as much as top 20%)
    stage1_model.addConstr(p40 - p80 >= EPSILON, name="P40_ge_P80")
    
    # Constraint 3: Combined lower/middle income benefit exceeds combined upper income
    # (P20 + P40-60) >= (P80-100 + P99)
    stage1_model.addConstr(
        p20 + p40 - p80 - p99 >= EPSILON,
        name="LowerMiddle_ge_Upper"
    )
    
    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    for group, idxs in ns_groups.items():
        stage1_model.addConstr(
            quicksum(x[i] for i in idxs) <= 1,
            name=f"NS_{group}_mutual_exclusivity"
        )
    
    # NS spending constraint
    # Total spending (negative revenue) from NS1-NS7 policies must be at least min_ns_spending
    # Note: Revenue is negative for spending, so we use <= -min_ns_spending
    stage1_model.addConstr(
        quicksum(x[i] * revenue[i] for i in ns_strict_indices) <= -min_ns_spending,
        name="MinimumNSSpending"
    )
    
    # Solve Stage 1
    stage1_model.optimize()
    
    # Check if model found a feasible solution
    if stage1_model.status != GRB.OPTIMAL:
        raise ValueError(
            f"Stage 1 optimization failed with status {stage1_model.status}. "
            f"The model may be infeasible - try reducing the NS spending requirement. "
            f"Current requirement: ${min_ns_spending:,}B"
        )
    
    gdp_star = stage1_model.ObjVal
    
    # === Stage 2: Maximize Revenue under optimal GDP ===
    if verbose:
        print("Running optimization (Stage 2: Maximize Revenue)...")
    
    stage2_model = Model("Stage2_MaximizeRevenue")
    if SUPPRESS_GUROBI_OUTPUT:
        stage2_model.setParam("OutputFlag", 0)
    
    # New decision variables for Stage 2
    x2 = stage2_model.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize dynamic revenue (break ties from Stage 1)
    stage2_model.setObjective(
        quicksum(x2[i] * revenue[i] for i in indices),
        GRB.MAXIMIZE
    )
    
    # Constraints (same as Stage 1, but with x2 variables)
    stage2_model.addConstr(
        quicksum(x2[i] * revenue[i] for i in indices) >= 0,
        name="RevenueNeutrality"
    )
    stage2_model.addConstr(
        quicksum(x2[i] * capital[i] for i in indices) >= 0,
        name="CapitalStock"
    )
    stage2_model.addConstr(
        quicksum(x2[i] * jobs[i] for i in indices) >= 0,
        name="Jobs"
    )
    stage2_model.addConstr(
        quicksum(x2[i] * wage[i] for i in indices) >= 0,
        name="WageRate"
    )
    
    # Equity constraints (same as Stage 1)
    p20 = quicksum(x2[i] * p20_arr[i] for i in indices)
    p40 = quicksum(x2[i] * p40_arr[i] for i in indices)
    p80 = quicksum(x2[i] * p80_arr[i] for i in indices)
    p99 = quicksum(x2[i] * p99_arr[i] for i in indices)
    
    stage2_model.addConstr(p20 - p40 <= DISTRIBUTIONAL_TOLERANCE, name="P20_P40_upper")
    stage2_model.addConstr(p40 - p20 <= DISTRIBUTIONAL_TOLERANCE, name="P40_P20_upper")
    stage2_model.addConstr(p20 - p99 >= EPSILON, name="P20_ge_P99")
    stage2_model.addConstr(p40 - p99 >= EPSILON, name="P40_ge_P99")
    stage2_model.addConstr(p20 - p80 >= EPSILON, name="P20_ge_P80")
    stage2_model.addConstr(p40 - p80 >= EPSILON, name="P40_ge_P80")
    stage2_model.addConstr(p20 + p40 - p80 - p99 >= EPSILON, name="LowerMiddle_ge_Upper")
    
    # NS constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        stage2_model.addConstr(
            quicksum(x2[i] for i in idxs) <= 1,
            name=f"NS_{group}_mutual_exclusivity"
        )
    
    stage2_model.addConstr(
        quicksum(x2[i] * revenue[i] for i in ns_strict_indices) <= -min_ns_spending,
        name="MinimumNSSpending"
    )
    
    # Additional constraint: Fix GDP to the optimal value from Stage 1
    # This ensures we don't sacrifice GDP to gain more revenue
    stage2_model.addConstr(
        quicksum(x2[i] * gdp[i] for i in indices) == gdp_star,
        name="GDPMatch"
    )
    
    # Solve Stage 2
    stage2_model.optimize()
    
    # Check if model found a feasible solution
    if stage2_model.status != GRB.OPTIMAL:
        raise ValueError(
            f"Stage 2 optimization failed with status {stage2_model.status}. "
            "This should not happen if Stage 1 succeeded."
        )
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
    selected_indices = [i for i in indices if x2[i].X > 0.5]
    selected_df = df.iloc[selected_indices].copy()
    
    return selected_df, gdp_star, stage2_model.ObjVal


def main() -> None:
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Optimize policy selection with national security and equity constraints.'
    )
    parser.add_argument(
        '--spending',
        type=int,
        default=DEFENSE_SPENDING["baseline"],
        help=f'Minimum NS spending in billions (default: {DEFENSE_SPENDING["baseline"]})'
    )
    args = parser.parse_args()
    
    # Load and clean data
    df, ns_groups = load_policy_data()
    ns_strict_indices = get_ns_strict_indices(df)
    
    # Run optimization with specified spending level
    result_df, gdp_impact, revenue_impact = optimize_policy_selection(
        df, ns_groups, ns_strict_indices, min_ns_spending=args.spending
    )
    
    # Display results
    display_results(result_df, gdp_impact, revenue_impact)
    
    # Save to CSV with spending level in filename
    output_file = f"max_gdp_defense{args.spending}.csv"
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'\n")


if __name__ == "__main__":
    main()