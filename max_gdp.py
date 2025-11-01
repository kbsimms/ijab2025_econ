"""
Policy Optimization Script - Basic GDP Maximization

WHAT THIS SCRIPT DOES:
This script finds the best combination of economic policies to maximize long-term
GDP growth while ensuring the government doesn't increase the deficit and national
security policies remain coherent.

Think of it as answering: "Which tax and spending policies should we adopt to grow
the economy as much as possible, without adding to the national debt, while making
sure our defense policies make sense together?"

HOW IT WORKS:
The script uses advanced mathematical optimization (linear programming) to evaluate
all possible combinations of policies and find the one that gives the highest GDP growth.

CONSTRAINTS (Requirements the solution must meet):
1. Revenue Neutrality: The total package can't increase the federal deficit
   - All selected policies together must generate at least $0 in revenue
   
2. National Security Coherence: Can't select conflicting defense policies
   - Example: Can't choose both "increase defense 10%" AND "increase defense 20%"
   - Only one option allowed per defense policy category (NS1, NS2, etc.)

OPTIMIZATION APPROACH:
Uses a two-stage process for finding the best solution:
- Stage 1: Find the maximum possible GDP growth
- Stage 2: Among all solutions with that maximum GDP, pick the one with the
           highest revenue surplus (extra fiscal room)

This ensures we get not just any solution, but the BEST solution.
"""

from typing import Tuple
import sys
import pandas as pd
from gurobipy import Model, GRB, quicksum, GurobiError

from config import COLUMNS, SUPPRESS_GUROBI_OUTPUT
from utils import load_policy_data, verify_ns_exclusivity, display_results
from logger import get_logger, LogLevel
from validation import ValidationError

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)


def optimize_policy_selection(
    df_clean: pd.DataFrame,
    ns_groups: dict,
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, float]:
    """
    Two-stage optimization to find optimal policy package.
    
    Stage 1: Maximize GDP subject to revenue neutrality and NS mutual exclusivity
        - Finds the maximum achievable GDP growth
        - Ensures total dynamic revenue is non-negative (revenue neutral or positive)
        - Enforces NS mutual exclusivity: at most one policy per NS group
        
    Stage 2: Maximize revenue while maintaining optimal GDP
        - Among all solutions that achieve the optimal GDP from Stage 1
        - Selects the one with the highest revenue surplus
        - This breaks ties when multiple policy combinations achieve the same GDP
    
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
    
    # === Stage 1: Maximize GDP subject to revenue constraint ===
    if verbose:
        logger.info("Running optimization (Stage 1: Maximize GDP)...")
    
    # Create Gurobi optimization model
    try:
        stage1_model = Model("Stage1_MaximizeGDP")
    except GurobiError as e:
        logger.error(f"Failed to create Gurobi model: {e}")
        logger.error("Please ensure Gurobi license is installed and valid.")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage1_model.setParam('OutputFlag', 0)
    
    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = stage1_model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Constraint: Total dynamic revenue must be non-negative (revenue neutral)
    # This ensures the policy package doesn't increase the deficit
    stage1_model.addConstr(
        quicksum(revenue[i] * x[i] for i in range(n)) >= 0,
        name="RevenueNeutrality"
    )
    
    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    if verbose and ns_groups:
        logger.info(f"  Adding {len(ns_groups)} NS mutual exclusivity constraints...")
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
    
    try:
        stage1_model.optimize()
    except GurobiError as e:
        logger.error(f"Stage 1 optimization failed: {e}")
        raise
    
    # Check optimization status
    if stage1_model.status != GRB.OPTIMAL:
        logger.error(f"Stage 1 did not find optimal solution. Status: {stage1_model.status}")
        if stage1_model.status == GRB.INFEASIBLE:
            logger.error("Model is infeasible. Constraints cannot be satisfied simultaneously.")
            logger.error("Possible causes:")
            logger.error("  - Revenue neutrality constraint too restrictive")
            logger.error("  - NS mutual exclusivity creates conflicts")
            logger.error("  - Data quality issues in Excel file")
        elif stage1_model.status == GRB.UNBOUNDED:
            logger.error("Model is unbounded. Objective can be improved indefinitely.")
        raise ValueError(f"Optimization failed with status {stage1_model.status}")
    
    # Store the optimal GDP value for use in Stage 2
    best_gdp = stage1_model.ObjVal
    logger.debug(f"Stage 1 optimal GDP: {best_gdp * 100:.4f}%")
    
    # === Stage 2: Maximize revenue while maintaining optimal GDP ===
    # This stage breaks ties when multiple solutions achieve the same GDP
    if verbose:
        logger.info("Running optimization (Stage 2: Maximize Revenue)...")
    
    try:
        stage2_model = Model("Stage2_MaximizeRevenue")
    except GurobiError as e:
        logger.error(f"Failed to create Stage 2 model: {e}")
        raise
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
    # This ensures we don't sacrifice GDP to gain more revenue
    stage2_model.addConstr(
        quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp,
        name="GDPMatch"
    )
    
    # NS mutual exclusivity constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        stage2_model.addConstr(
            quicksum(x2[i] for i in idxs) <= 1,
            name=f"NS_{group}_mutual_exclusivity"
        )
    
    # Objective: Maximize revenue surplus (among optimal GDP solutions)
    stage2_model.setObjective(
        quicksum(revenue[i] * x2[i] for i in range(n)),
        GRB.MAXIMIZE
    )
    
    try:
        stage2_model.optimize()
    except GurobiError as e:
        logger.error(f"Stage 2 optimization failed: {e}")
        raise
    
    # Check optimization status
    if stage2_model.status != GRB.OPTIMAL:
        logger.error(f"Stage 2 did not find optimal solution. Status: {stage2_model.status}")
        raise ValueError(
            f"Stage 2 optimization failed with status {stage2_model.status}. "
            "This should not happen if Stage 1 succeeded."
        )
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
    selected_indices = [i for i in range(n) if x2[i].X > 0.5]
    selected_df = df_clean.iloc[selected_indices].copy()
    
    # Verify NS mutual exclusivity in solution
    verify_ns_exclusivity(df_clean, ns_groups, selected_indices, verbose)
    
    return selected_df, best_gdp, stage2_model.ObjVal


def main() -> None:
    """Main execution function."""
    try:
        # Load and clean data
        logger.info("Starting GDP maximization optimization...")
        df_clean, ns_groups = load_policy_data()
        
        # Run optimization
        result_df, gdp_impact, revenue_impact = optimize_policy_selection(
            df_clean, ns_groups
        )
        
        # Display results
        display_results(result_df, gdp_impact, revenue_impact)
        
        # Save to CSV
        output_file = "max_gdp.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"✓ Results saved to '{output_file}'")
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except GurobiError as e:
        logger.error(f"Gurobi optimization error: {e}")
        logger.error("Please check your Gurobi license and model formulation.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
