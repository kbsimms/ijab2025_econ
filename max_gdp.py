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
1. Revenue Surplus Requirement: The total package must generate substantial surplus
   - All selected policies together must generate at least $600B in revenue

2. National Security Coherence: Can't select conflicting defense policies
   - Example: Can't choose both "increase defense 10%" AND "increase defense 20%"
   - Only one option allowed per defense policy category (NS1, NS2, etc.)

OPTIMIZATION APPROACH:
Uses a three-stage process for finding the best solution:
- Stage 1: Find the maximum possible GDP growth
- Stage 2: Among all solutions with that maximum GDP, pick those that
           create the most jobs (maximizes employment)
- Stage 3: Among all solutions with maximum GDP and maximum jobs, pick the one
           with the highest revenue surplus (maximizes fiscal conservatism)

This ensures we get not just any solution, but the BEST solution.
"""

import sys
import traceback

from gurobipy import GRB, GurobiError, Model, quicksum
import pandas as pd

from config import COLUMNS, REVENUE_SURPLUS_REQUIREMENT, SUPPRESS_GUROBI_OUTPUT
from logger import LogLevel, get_logger
from utils import display_results, load_policy_data, verify_ns_exclusivity
from validation import ValidationError

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)

# Binary variable decision threshold
BINARY_THRESHOLD = 0.5


def optimize_policy_selection(  # noqa: PLR0912, PLR0915
    df_clean: pd.DataFrame, ns_groups: dict[str, list[int]], verbose: bool = True
) -> tuple[pd.DataFrame, float, float, float]:
    """
    Three-stage optimization to find optimal policy package.

    Stage 1: Maximize GDP subject to revenue surplus requirement and NS mutual exclusivity
        - Finds the maximum achievable GDP growth
        - Ensures total dynamic revenue is at least $600B (substantial surplus)
        - Enforces NS mutual exclusivity: at most one policy per NS group

    Stage 2: Maximize job creation while maintaining optimal GDP
        - Among all solutions that achieve the optimal GDP from Stage 1
        - Selects those that create the most jobs
        - Reduces solution space but may still have multiple alternatives

    Stage 3: Maximize revenue surplus while maintaining optimal GDP and jobs
        - Among all solutions with max GDP and max jobs from Stages 1 & 2
        - Selects the one with highest revenue surplus
        - Provides final tiebreaker for unique solution

    National Security (NS) Mutual Exclusivity:
        For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
        at most one option can be selected. This prevents selecting
        conflicting national security policies within the same category.

    Args:
        df_clean: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        verbose: If True, prints progress messages

    Returns:
        tuple: (selected_df, gdp_impact, jobs_impact, revenue_impact)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - jobs_impact: Total jobs created
            - revenue_impact: Total revenue surplus achieved
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
    except GurobiError:
        logger.exception(
            "Failed to create Gurobi model. Please ensure Gurobi license is installed and valid."
        )
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage1_model.setParam("OutputFlag", 0)

    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = stage1_model.addVars(n, vtype=GRB.BINARY, name="x")

    # Constraint: Total dynamic revenue must be at least $600B (revenue surplus)
    # This ensures the policy package generates substantial revenue surplus
    stage1_model.addConstr(
        quicksum(revenue[i] * x[i] for i in range(n)) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )

    # NS mutual exclusivity constraints
    # For each NS group (e.g., NS1 with options NS1A, NS1B, NS1C),
    # at most one option can be selected
    if verbose and ns_groups:
        logger.info(f"  Adding {len(ns_groups)} NS mutual exclusivity constraints...")
    for group, idxs in ns_groups.items():
        stage1_model.addConstr(
            quicksum(x[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity"
        )

    # Objective: Maximize total GDP impact
    stage1_model.setObjective(quicksum(gdp[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

    try:
        stage1_model.optimize()
    except GurobiError:
        logger.exception("Stage 1 optimization failed")
        raise

    # Check optimization status
    if stage1_model.status != GRB.OPTIMAL:
        logger.error(f"Stage 1 did not find optimal solution. Status: {stage1_model.status}")
        if stage1_model.status == GRB.INFEASIBLE:
            logger.error("Model is infeasible. Constraints cannot be satisfied simultaneously.")
            logger.error("Possible causes:")
            logger.error("  - Revenue surplus requirement ($600B) too restrictive")
            logger.error("  - NS mutual exclusivity creates conflicts")
            logger.error("  - Data quality issues in Excel file")
        elif stage1_model.status == GRB.UNBOUNDED:
            logger.error("Model is unbounded. Objective can be improved indefinitely.")
        raise ValueError(f"Optimization failed with status {stage1_model.status}")

    # Store the optimal GDP value for use in Stage 2
    best_gdp = stage1_model.ObjVal
    logger.debug(f"Stage 1 optimal GDP: {best_gdp * 100:.4f}%")

    # === Stage 2: Maximize job creation while maintaining optimal GDP ===
    # This stage breaks ties when multiple solutions achieve the same GDP
    if verbose:
        logger.info("Running optimization (Stage 2: Maximize Jobs)...")

    try:
        stage2_model = Model("Stage2_MaximizeRevenue")
    except GurobiError:
        logger.exception("Failed to create Stage 2 model")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage2_model.setParam("OutputFlag", 0)

    # New decision variables for Stage 2
    x2 = stage2_model.addVars(n, vtype=GRB.BINARY, name="x")

    # Extract jobs data for Stage 2 objective
    jobs = df_clean[COLUMNS["jobs"]].values

    # Constraint: Revenue surplus requirement (same as Stage 1)
    stage2_model.addConstr(
        quicksum(revenue[i] * x2[i] for i in range(n)) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )

    # Constraint: Must achieve exactly the optimal GDP from Stage 1
    # This ensures we don't sacrifice GDP to gain more revenue
    stage2_model.addConstr(quicksum(gdp[i] * x2[i] for i in range(n)) == best_gdp, name="GDPMatch")

    # NS mutual exclusivity constraints (same as Stage 1)
    for group, idxs in ns_groups.items():
        stage2_model.addConstr(
            quicksum(x2[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity"
        )

    # Objective: Maximize job creation (among optimal GDP solutions)
    stage2_model.setObjective(quicksum(jobs[i] * x2[i] for i in range(n)), GRB.MAXIMIZE)

    try:
        stage2_model.optimize()
    except GurobiError:
        logger.exception("Stage 2 optimization failed")
        raise

    # Check optimization status
    if stage2_model.status != GRB.OPTIMAL:
        logger.error(f"Stage 2 did not find optimal solution. Status: {stage2_model.status}")
        raise ValueError(
            f"Stage 2 optimization failed with status {stage2_model.status}. "
            "This should not happen if Stage 1 succeeded."
        )

    # Extract solution: policies where x2[i] > BINARY_THRESHOLD are selected
    # Using BINARY_THRESHOLD handles numerical precision issues in binary variables
    selected_indices = [i for i in range(n) if x2[i].X > BINARY_THRESHOLD]
    selected_df = df_clean.iloc[selected_indices].copy()

    # Verify NS mutual exclusivity in solution
    verify_ns_exclusivity(df_clean, ns_groups, selected_indices, verbose)

    # Store the optimal jobs value for use in Stage 3
    best_jobs = stage2_model.ObjVal
    logger.debug(f"Stage 2 optimal jobs: {best_jobs:,.0f}")

    # === Stage 3: Maximize revenue surplus while maintaining optimal GDP and jobs ===
    if verbose:
        logger.info("Running optimization (Stage 3: Maximize Revenue)...")

    try:
        stage3_model = Model("Stage3_MaximizeRevenue")
    except GurobiError:
        logger.exception("Failed to create Stage 3 model")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage3_model.setParam("OutputFlag", 0)

    # New decision variables for Stage 3
    x3 = stage3_model.addVars(n, vtype=GRB.BINARY, name="x")

    # Constraint: Revenue surplus requirement (same as Stage 1)
    stage3_model.addConstr(
        quicksum(revenue[i] * x3[i] for i in range(n)) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )

    # Constraint: Must achieve exactly the optimal GDP from Stage 1
    stage3_model.addConstr(quicksum(gdp[i] * x3[i] for i in range(n)) == best_gdp, name="GDPMatch")

    # Constraint: Must achieve exactly the optimal jobs from Stage 2
    stage3_model.addConstr(
        quicksum(jobs[i] * x3[i] for i in range(n)) == best_jobs, name="JobsMatch"
    )

    # NS mutual exclusivity constraints (same as previous stages)
    for group, idxs in ns_groups.items():
        stage3_model.addConstr(
            quicksum(x3[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity"
        )

    # Objective: Maximize revenue surplus (final tiebreaker)
    stage3_model.setObjective(quicksum(revenue[i] * x3[i] for i in range(n)), GRB.MAXIMIZE)

    try:
        stage3_model.optimize()
    except GurobiError:
        logger.exception("Stage 3 optimization failed")
        raise

    # Check optimization status
    if stage3_model.status != GRB.OPTIMAL:
        logger.error(f"Stage 3 did not find optimal solution. Status: {stage3_model.status}")
        raise ValueError(
            f"Stage 3 optimization failed with status {stage3_model.status}. "
            "This should not happen if Stage 2 succeeded."
        )

    # Extract final solution
    selected_indices = [i for i in range(n) if x3[i].X > BINARY_THRESHOLD]
    selected_df = df_clean.iloc[selected_indices].copy()

    # Verify NS mutual exclusivity in solution
    verify_ns_exclusivity(df_clean, ns_groups, selected_indices, verbose)

    # Return all three optimized values
    revenue_impact = stage3_model.ObjVal
    return selected_df, best_gdp, best_jobs, revenue_impact


def main() -> None:
    """Main execution function."""
    try:
        # Load and clean data
        logger.info("Starting GDP maximization optimization...")
        df_clean, ns_groups = load_policy_data()

        # Run optimization (now returns 4 values including revenue from Stage 3)
        result_df, gdp_impact, jobs_impact, revenue_impact = optimize_policy_selection(
            df_clean, ns_groups
        )

        # Display results
        display_results(result_df, gdp_impact, revenue_impact)

        logger.info("\n[Three-Stage Optimization Results]")
        logger.info(f"  Stage 1 - Optimal GDP: {gdp_impact * 100:.4f}%")
        logger.info(f"  Stage 2 - Optimal Jobs: {jobs_impact:,.0f}")
        logger.info(f"  Stage 3 - Optimal Revenue: ${revenue_impact:,.2f}B")

        # Save to CSV
        output_file = "max_gdp.csv"
        result_df.to_csv(output_file, index=False)
        logger.info(f"[OK] Results saved to '{output_file}'")

    except ValidationError:
        logger.exception("Validation error")
        sys.exit(1)
    except GurobiError:
        logger.exception(
            "Gurobi optimization error. Please check your Gurobi license and model formulation."
        )
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
