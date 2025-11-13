"""
Policy Portfolio Optimization Script with Progressive Distribution Constraints

WHAT THIS SCRIPT DOES:
This script finds the best combination of economic policies to maximize GDP growth
while ensuring:
- Specific required policies are included (S14, 28, 29, 42, S5, 54, S9)
- Specific policies are prohibited (5, 19, 64, 63) + excluded policies (37, 43, 49, 68)
- Progressive distribution: lower/middle income groups benefit MORE than high earners
  (without requiring all groups to have positive changes)
- The government doesn't increase the deficit
- Economic outcomes are positive (jobs, wages, capital all improve)
- National defense gets adequate funding
- Policy choices are coherent (no contradictory policies selected together)

Think of it as answering: "Which policies grow the economy the most while being
fair to working families, maintaining strong defense, and not adding to the debt,
with specific progressive policies required?"

CONSTRAINTS (Requirements every solution must meet):

1. FISCAL: Revenue surplus of at least $600B - generates substantial surplus

2. ECONOMIC: Must improve the economy across the board
   - More jobs created
   - Higher wages
   - More capital investment

3. REQUIRED POLICIES (MUST include):
   - S14: Extend expanded ACA premiums & tax credits from the American Rescue Plan
   - 28: Tax Carried Interest as Ordinary Income
   - 29: Raise the Top Capital Gains and Dividend Tax Rate to 30%
   - 42: Eliminate 1031 Like-Kind Exchanges
   - S5: Two-years of subsidized tuition for students from families earning less than $125,000
   - 54: Make Child Tax Credit First Dollar Refundable
   - S9: Create a national comprehensive paid family and medical leave program

4. PROHIBITED POLICIES (CANNOT include):
   - 5: Lower the Top Rate on Capital Gains and Dividends to 15 Percent
   - 19: Eliminate the Child Tax Credit
   - 64: Eliminate the Net Investment Income Tax
   - 63: Implement $2,500 per Year Roth-style Universal Savings Accounts
   - 37: Corporate Surtax of 5% (from EXCLUDED_POLICIES)
   - 43: Enact a 5% VAT (from EXCLUDED_POLICIES)
   - 49: Reinstate the Cadillac Tax (from EXCLUDED_POLICIES)
   - 68: Replace CIT with 5% VAT (from EXCLUDED_POLICIES)

5. PROGRESSIVE DISTRIBUTIONAL CONSTRAINT:
   - P20 (bottom 20%) must benefit MORE than P80-100 (top 20%)
   - P20 (bottom 20%) must benefit MORE than P99 (top 1%)
   - P40-60 (middle class) must benefit MORE than P80-100 (top 20%)
   - P40-60 (middle class) must benefit MORE than P99 (top 1%)
   Note: Changes do NOT need to be positive, just progressive

6. POLICY COHERENCE: Can't select contradictory policies
   - 15 mutually exclusive groups (e.g., can't have two different corporate tax rates)

7. NATIONAL SECURITY: Adequate defense funding
   - Can specify exact defense spending requirement (e.g., $3,000B)
   - Only one option per defense category

OPTIMIZATION APPROACH:
Three-stage process ensures the very best solution:
- Stage 1: Find maximum possible GDP growth meeting all constraints
- Stage 2: Among all max-GDP solutions, pick those that create the most jobs
- Stage 3: Among solutions with max GDP and max jobs, pick one with highest revenue surplus

This hierarchical approach ensures optimal GDP, employment, and fiscal outcomes.

USAGE EXAMPLES:
    # Run full analysis across all defense spending levels
    python policy_portfolio_optimization.py

    # Run single optimization with $3,000B defense spending
    python policy_portfolio_optimization.py --spending 3000

    # Run full range (same as default, but explicit)
    python policy_portfolio_optimization.py --all

OUTPUTS:
- CSV files with selected policies for each spending level
- Console output with comprehensive results for each level
- Summary files with policy decisions and economic effects
"""

import argparse
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from gurobipy import GRB, GurobiError, Model, quicksum
import pandas as pd

from config import (
    COLUMNS,
    DEFENSE_SPENDING,
    EPSILON,
    EXCLUDED_POLICIES,
    POLICY_CO_EXCLUSIONS,
    REVENUE_SURPLUS_REQUIREMENT,
    SPENDING_RANGE,
    SUPPRESS_GUROBI_OUTPUT,
)
from logger import LogLevel, get_logger, set_global_level
from optimizer_utils import get_policy_indices_by_codes
from utils import display_results, get_ns_strict_indices, load_policy_data
from validation import (
    ValidationError,
    validate_optimization_inputs,
    validate_output_directory,
    validate_spending_level,
)

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)

# Binary variable decision threshold
BINARY_THRESHOLD = 0.5

# Required policies (MUST be included)
REQUIRED_POLICIES = ["S14", "28", "29", "42", "S5", "54", "S9"]

# Additional prohibited policies (in addition to EXCLUDED_POLICIES from config)
ADDITIONAL_PROHIBITED_POLICIES = ["5", "19", "64", "63"]


def define_policy_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Define mutually exclusive policy groups.

    Returns:
        Dictionary mapping group names to lists of policy indices
    """
    policy_groups: dict[str, list[int]] = {}

    # 1. Corporate Tax Rate/Structure
    policy_groups["corporate_tax"] = get_policy_indices_by_codes(df, ["11", "36", "68"])

    # 2. Gas Tax Increases
    policy_groups["gas_tax"] = get_policy_indices_by_codes(df, ["47", "48"])

    # 3. Estate Tax
    policy_groups["estate_tax"] = get_policy_indices_by_codes(df, ["12", "44", "46", "69"])

    # 4. Child Tax Credit - Refundability
    policy_groups["ctc_refundability"] = get_policy_indices_by_codes(df, ["53", "54"])

    # 5. Social Security Payroll Tax Cap
    policy_groups["ss_payroll_cap"] = get_policy_indices_by_codes(df, ["34", "35"])

    # 6. Payroll Tax Rate Changes
    policy_groups["payroll_rate"] = get_policy_indices_by_codes(df, ["4", "33"])

    # 7. EITC/CDCTC Reforms
    policy_groups["eitc_reforms"] = get_policy_indices_by_codes(df, ["21", "51", "52", "55", "S15"])

    # 8. Individual Income Tax Structure
    policy_groups["individual_tax_structure"] = get_policy_indices_by_codes(
        df, ["1", "2", "3", "14", "59"]
    )

    # 9. Child Tax Credit - Comprehensive
    policy_groups["ctc_comprehensive"] = get_policy_indices_by_codes(df, ["19", "20", "55", "S13"])

    # 10. Section 199A Deduction
    policy_groups["section_199a"] = get_policy_indices_by_codes(df, ["10", "38"])

    # 11. Home Mortgage Interest Deduction
    policy_groups["mortgage_deduction"] = get_policy_indices_by_codes(df, ["23", "24"])

    # 12. Charitable Deduction
    policy_groups["charitable_deduction"] = get_policy_indices_by_codes(df, ["25", "58"])

    # 13. Capital Gains Tax Rate
    policy_groups["capital_gains"] = get_policy_indices_by_codes(df, ["5", "29", "30"])

    # 14. Depreciation/Expensing
    policy_groups["depreciation"] = get_policy_indices_by_codes(df, ["7", "40", "65"])

    # 15. Value Added Tax (VAT)
    policy_groups["vat"] = get_policy_indices_by_codes(df, ["43", "68"])

    # Remove empty groups
    return {k: v for k, v in policy_groups.items() if len(v) > 0}


def add_progressive_distribution_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    p20_arr: Any,  # ArrayLike
    p40_arr: Any,  # ArrayLike
    p80_arr: Any,  # ArrayLike
    p99_arr: Any,  # ArrayLike
    indices: range,
    epsilon: float = EPSILON,
    logger_obj: Any = None,
) -> None:
    """
    Add progressive distribution constraints WITHOUT requiring positive changes.

    This ensures that lower/middle income groups benefit MORE than high earners,
    but does NOT require that all groups have positive changes.

    Constraints:
    - Δ After-Tax Income(P20) > Δ After-Tax Income(P80-100)
    - Δ After-Tax Income(P20) > Δ After-Tax Income(P99)
    - Δ After-Tax Income(P40-60) > Δ After-Tax Income(P80-100)
    - Δ After-Tax Income(P40-60) > Δ After-Tax Income(P99)

    Args:
        model: Gurobi model
        x: Decision variables
        p20_arr: P20 after-tax income effects
        p40_arr: P40-60 after-tax income effects
        p80_arr: P80-100 after-tax income effects
        p99_arr: P99 after-tax income effects
        indices: Range of policy indices
        epsilon: Small value to ensure strict inequality
        logger_obj: Optional logger
    """
    # Calculate total after-tax income change for each percentile group
    p20 = quicksum(x[i] * p20_arr[i] for i in indices)
    p40 = quicksum(x[i] * p40_arr[i] for i in indices)
    p80 = quicksum(x[i] * p80_arr[i] for i in indices)
    p99 = quicksum(x[i] * p99_arr[i] for i in indices)

    # Progressive distribution: Lower/middle income groups must benefit
    # MORE than upper groups (with epsilon for strict inequality)
    # Note: No requirement for positive changes, just that P20 > P80-100 and P20 > P99
    model.addConstr(p20 - p80 >= epsilon, name="P20_gt_P80")
    model.addConstr(p20 - p99 >= epsilon, name="P20_gt_P99")
    model.addConstr(p40 - p80 >= epsilon, name="P40_gt_P80")
    model.addConstr(p40 - p99 >= epsilon, name="P40_gt_P99")

    if logger_obj:
        logger_obj.debug(
            "Added progressive distribution constraints (without positivity requirement)"
        )


def add_required_policy_constraints(
    model: Model,
    x: Any,  # Gurobi tupledict[int, Var]
    df: pd.DataFrame,
    required_codes: list[str],
    logger_obj: Any = None,
) -> int:
    """
    Add constraints to require certain policies to be selected.

    Args:
        model: Gurobi model to add constraints to
        x: Decision variables dict (policy index -> variable)
        df: DataFrame containing policy data
        required_codes: List of policy codes that must be included
        logger_obj: Optional logger for progress messages

    Returns:
        Number of constraints added
    """
    required_indices = get_policy_indices_by_codes(df, required_codes)

    for idx in required_indices:
        model.addConstr(x[idx] == 1, name=f"Required_policy_{idx}")

    if logger_obj and len(required_indices) > 0:
        logger_obj.debug(f"Added {len(required_indices)} required policy constraints")

    return len(required_indices)


def optimize_policy_selection(  # noqa: PLR0912, PLR0915
    df: pd.DataFrame,
    ns_groups: dict[str, list[int]],
    ns_strict_indices: list[int],
    min_ns_spending: int = DEFENSE_SPENDING["baseline"],
    verbose: bool = True,
) -> tuple[pd.DataFrame, float, float, float, dict[str, float]]:
    """
    Three-stage optimization with required/prohibited policies, progressive distribution, and NS constraints.

    Stage 1: Maximize GDP subject to all constraints
        - Fiscal: Revenue surplus requirement (sum of dynamic revenue >= 600)
        - Economic: Non-negative capital stock, jobs, wage rate
        - Progressive distribution: P20, P40-60 must benefit MORE than P80-100, P99
        - Required policies: Must include S14, 28, 29, 42, S5, 54, S9
        - Prohibited policies: Cannot include 5, 19, 64, 63, plus excluded (37, 43, 49, 68)
        - Policy mutual exclusivity: At most one policy per competing group (15 groups)
        - Policy co-exclusions: Special exclusion rules (e.g., if 68 then not 37)
        - NS mutual exclusivity: At most one policy per NS group
        - NS spending target: Exactly min_ns_spending on NS1-NS7 policies

    Stage 2: Maximize job creation while maintaining optimal GDP
        - Same constraints as Stage 1
        - Additionally constrains GDP to equal the optimal value from Stage 1

    Stage 3: Maximize revenue surplus while maintaining optimal GDP and jobs
        - Same constraints as Stage 1
        - Additionally constrains GDP and jobs to optimal values from Stages 1 & 2

    Progressive Distribution (Key Difference from max_gdp_defense.py):
        - P20, P40-60 must benefit MORE than P80-100, P99
        - Does NOT require all groups to have positive changes (no >= 0 constraint)
        - Allows everyone to potentially be worse off, as long as it's progressive

    Args:
        df: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        ns_strict_indices: List of indices for NS1-NS7 policies
        min_ns_spending: Minimum NS spending in billions (default: 3000)
        verbose: If True, prints progress messages

    Returns:
        tuple: (selected_df, gdp_impact, jobs_impact, revenue_impact, kpi_dict)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - jobs_impact: Total jobs created
            - revenue_impact: Total revenue surplus achieved
            - kpi_dict: Dictionary of all KPI values
    """
    # Extract data arrays from DataFrame for optimization
    n = len(df)
    indices = range(n)

    # Convert DataFrame columns to numpy arrays for efficient access
    gdp = df[COLUMNS["gdp"]].values
    revenue = df[COLUMNS["dynamic_revenue"]].values
    jobs = df[COLUMNS["jobs"]].values
    capital = df[COLUMNS["capital"]].values
    wage = df[COLUMNS["wage"]].values
    p20_arr = df[COLUMNS["p20"]].values
    p40_arr = df[COLUMNS["p40_60"]].values
    p80_arr = df[COLUMNS["p80_100"]].values
    p99_arr = df[COLUMNS["p99"]].values

    # Get indices for required and prohibited policies
    required_indices = get_policy_indices_by_codes(df, REQUIRED_POLICIES)
    # Combine additional prohibited with EXCLUDED_POLICIES from config
    all_prohibited = list(set(ADDITIONAL_PROHIBITED_POLICIES + EXCLUDED_POLICIES))
    prohibited_indices = get_policy_indices_by_codes(df, all_prohibited)

    # Validate inputs before optimization
    try:
        validate_optimization_inputs(df, ns_groups, ns_strict_indices, min_ns_spending)
    except ValidationError:
        if verbose:
            logger.exception("Input validation failed")
        raise

    if verbose:
        logger.info("Policy Portfolio Optimization")
        logger.info(f"  NS spending requirement: ${min_ns_spending:,}B")
        logger.info(f"  Required policies: {len(required_indices)} policies must be included")
        for code in REQUIRED_POLICIES:
            idx_list = get_policy_indices_by_codes(df, [code])
            if idx_list:
                policy_name = df.iloc[idx_list[0]][COLUMNS["option"]]
                logger.info(f"    - {policy_name}")

        logger.info(f"  Prohibited policies: {len(prohibited_indices)} policies cannot be included")
        for code in all_prohibited:
            idx_list = get_policy_indices_by_codes(df, [code])
            if idx_list:
                policy_name = df.iloc[idx_list[0]][COLUMNS["option"]]
                logger.info(f"    - {policy_name}")

    # === Stage 1: Maximize GDP ===
    if verbose:
        logger.info("\nRunning optimization (Stage 1: Maximize GDP)...")

    try:
        stage1_model = Model("Stage1_MaximizeGDP")
    except GurobiError:
        if verbose:
            logger.exception("Failed to create Gurobi model")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage1_model.setParam("OutputFlag", 0)

    # Decision variables: x[i] = 1 if policy i is selected, 0 otherwise
    x = stage1_model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Objective: Maximize total GDP impact
    stage1_model.setObjective(quicksum(x[i] * gdp[i] for i in indices), GRB.MAXIMIZE)

    # === Constraints ===

    # 1. FISCAL: Revenue surplus requirement
    stage1_model.addConstr(
        quicksum(x[i] * revenue[i] for i in indices) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )

    # 2. ECONOMIC: Non-negative impacts
    stage1_model.addConstr(quicksum(x[i] * capital[i] for i in indices) >= 0, name="CapitalStock")
    stage1_model.addConstr(quicksum(x[i] * jobs[i] for i in indices) >= 0, name="Jobs")
    stage1_model.addConstr(quicksum(x[i] * wage[i] for i in indices) >= 0, name="WageRate")

    # 3. REQUIRED POLICIES: Must be included
    add_required_policy_constraints(
        stage1_model, x, df, REQUIRED_POLICIES, logger_obj=logger if verbose else None
    )

    # 4. PROHIBITED POLICIES: Cannot be included (excluded + additional)
    for idx in prohibited_indices:
        stage1_model.addConstr(x[idx] == 0, name=f"Prohibited_policy_{idx}")

    if verbose:
        logger.debug(f"Added {len(prohibited_indices)} prohibited policy constraints")

    # 5. POLICY CO-EXCLUSIONS: Special exclusion rules
    for code_a, code_b in POLICY_CO_EXCLUSIONS:
        idx_a = get_policy_indices_by_codes(df, [code_a])
        idx_b = get_policy_indices_by_codes(df, [code_b])
        if len(idx_a) > 0 and len(idx_b) > 0:
            stage1_model.addConstr(
                x[idx_a[0]] + x[idx_b[0]] <= 1, name=f"Policy_{code_a}_excludes_{code_b}"
            )

    # 6. PROGRESSIVE DISTRIBUTION: Lower/middle must benefit MORE than wealthy
    # (WITHOUT requiring positive changes for anyone)
    add_progressive_distribution_constraints(
        stage1_model,
        x,
        p20_arr,
        p40_arr,
        p80_arr,
        p99_arr,
        indices,
        logger_obj=logger if verbose else None,
    )

    # 7. POLICY MUTUAL EXCLUSIVITY
    policy_groups = define_policy_groups(df)
    count = 0
    for group_name, idxs in policy_groups.items():
        if len(idxs) > 1:
            stage1_model.addConstr(
                quicksum(x[i] for i in idxs) <= 1, name=f"Policy_{group_name}_mutual_exclusivity"
            )
            count += 1

    if verbose:
        logger.debug(f"Added {count} policy mutual exclusivity constraints")

    # 8. NS MUTUAL EXCLUSIVITY
    for group, idxs in ns_groups.items():
        stage1_model.addConstr(
            quicksum(x[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity"
        )

    # 9. NS SPENDING CONSTRAINT: Exact spending on NS1-NS7 policies
    stage1_model.addConstr(
        quicksum(x[i] * revenue[i] for i in ns_strict_indices) == -min_ns_spending,
        name="ExactNSSpending",
    )

    if verbose:
        logger.debug(f"Added NS spending constraint: ${min_ns_spending:,}B")

    # Solve Stage 1
    try:
        stage1_model.optimize()
    except GurobiError:
        if verbose:
            logger.exception("Stage 1 optimization failed")
        raise

    # Check if model found a feasible solution
    if stage1_model.status != GRB.OPTIMAL:
        error_msg = f"Stage 1 optimization failed with status {stage1_model.status}"
        if verbose:
            logger.error(error_msg)
            if stage1_model.status == GRB.INFEASIBLE:
                logger.error("Model is infeasible. Possible causes:")
                logger.error(f"  - NS spending requirement too restrictive: ${min_ns_spending:,}B")
                logger.error("  - Required policies conflict with other constraints")
                logger.error("  - Progressive distribution cannot be satisfied")
                logger.error("  - Revenue surplus requirement too restrictive")
        raise ValueError(
            f"{error_msg}. "
            f"The model may be infeasible - try different NS spending or policy requirements. "
            f"Current NS requirement: ${min_ns_spending:,}B"
        )

    gdp_star = stage1_model.ObjVal
    if verbose:
        logger.debug(f"Stage 1 optimal GDP: {gdp_star * 100:.4f}%")

    # === Stage 2: Maximize Job Creation under optimal GDP ===
    if verbose:
        logger.info("Running optimization (Stage 2: Maximize Jobs)...")

    try:
        stage2_model = Model("Stage2_MaximizeJobs")
    except GurobiError:
        if verbose:
            logger.exception("Failed to create Stage 2 model")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage2_model.setParam("OutputFlag", 0)

    # New decision variables for Stage 2
    x2 = stage2_model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Objective: Maximize job creation
    stage2_model.setObjective(quicksum(x2[i] * jobs[i] for i in indices), GRB.MAXIMIZE)

    # Add all constraints from Stage 1
    stage2_model.addConstr(
        quicksum(x2[i] * revenue[i] for i in indices) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )
    stage2_model.addConstr(quicksum(x2[i] * capital[i] for i in indices) >= 0, name="CapitalStock")
    stage2_model.addConstr(quicksum(x2[i] * jobs[i] for i in indices) >= 0, name="Jobs")
    stage2_model.addConstr(quicksum(x2[i] * wage[i] for i in indices) >= 0, name="WageRate")

    add_required_policy_constraints(stage2_model, x2, df, REQUIRED_POLICIES)

    for idx in prohibited_indices:
        stage2_model.addConstr(x2[idx] == 0, name=f"Prohibited_policy_{idx}")

    for code_a, code_b in POLICY_CO_EXCLUSIONS:
        idx_a = get_policy_indices_by_codes(df, [code_a])
        idx_b = get_policy_indices_by_codes(df, [code_b])
        if len(idx_a) > 0 and len(idx_b) > 0:
            stage2_model.addConstr(
                x2[idx_a[0]] + x2[idx_b[0]] <= 1, name=f"Policy_{code_a}_excludes_{code_b}"
            )

    add_progressive_distribution_constraints(
        stage2_model, x2, p20_arr, p40_arr, p80_arr, p99_arr, indices
    )

    for group_name, idxs in policy_groups.items():
        if len(idxs) > 1:
            stage2_model.addConstr(
                quicksum(x2[i] for i in idxs) <= 1, name=f"Policy_{group_name}_mutual_exclusivity"
            )

    for group, idxs in ns_groups.items():
        stage2_model.addConstr(
            quicksum(x2[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity"
        )

    stage2_model.addConstr(
        quicksum(x2[i] * revenue[i] for i in ns_strict_indices) == -min_ns_spending,
        name="ExactNSSpending",
    )

    # Additional constraint: Fix GDP to optimal value from Stage 1
    stage2_model.addConstr(quicksum(x2[i] * gdp[i] for i in indices) == gdp_star, name="GDPMatch")

    # Solve Stage 2
    try:
        stage2_model.optimize()
    except GurobiError:
        if verbose:
            logger.exception("Stage 2 optimization failed")
        raise

    if stage2_model.status != GRB.OPTIMAL:
        error_msg = f"Stage 2 optimization failed with status {stage2_model.status}"
        if verbose:
            logger.error(error_msg)
        raise ValueError(f"{error_msg}. This should not happen if Stage 1 succeeded.")

    jobs_star = stage2_model.ObjVal
    if verbose:
        logger.debug(f"Stage 2 optimal jobs: {jobs_star:,.0f}")

    # === Stage 3: Maximize Revenue under optimal GDP and Jobs ===
    if verbose:
        logger.info("Running optimization (Stage 3: Maximize Revenue)...")

    try:
        stage3_model = Model("Stage3_MaximizeRevenue")
    except GurobiError:
        if verbose:
            logger.exception("Failed to create Stage 3 model")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage3_model.setParam("OutputFlag", 0)

    # New decision variables for Stage 3
    x3 = stage3_model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Objective: Maximize revenue surplus
    stage3_model.setObjective(quicksum(x3[i] * revenue[i] for i in indices), GRB.MAXIMIZE)

    # Add all constraints from previous stages
    stage3_model.addConstr(
        quicksum(x3[i] * revenue[i] for i in indices) >= REVENUE_SURPLUS_REQUIREMENT,
        name="RevenueSurplus",
    )
    stage3_model.addConstr(quicksum(x3[i] * capital[i] for i in indices) >= 0, name="CapitalStock")
    stage3_model.addConstr(quicksum(x3[i] * jobs[i] for i in indices) >= 0, name="Jobs")
    stage3_model.addConstr(quicksum(x3[i] * wage[i] for i in indices) >= 0, name="WageRate")

    add_required_policy_constraints(stage3_model, x3, df, REQUIRED_POLICIES)

    for idx in prohibited_indices:
        stage3_model.addConstr(x3[idx] == 0, name=f"Prohibited_policy_{idx}")

    for code_a, code_b in POLICY_CO_EXCLUSIONS:
        idx_a = get_policy_indices_by_codes(df, [code_a])
        idx_b = get_policy_indices_by_codes(df, [code_b])
        if len(idx_a) > 0 and len(idx_b) > 0:
            stage3_model.addConstr(
                x3[idx_a[0]] + x3[idx_b[0]] <= 1, name=f"Policy_{code_a}_excludes_{code_b}"
            )

    add_progressive_distribution_constraints(
        stage3_model, x3, p20_arr, p40_arr, p80_arr, p99_arr, indices
    )

    for group_name, idxs in policy_groups.items():
        if len(idxs) > 1:
            stage3_model.addConstr(
                quicksum(x3[i] for i in idxs) <= 1, name=f"Policy_{group_name}_mutual_exclusivity"
            )

    for group, idxs in ns_groups.items():
        stage3_model.addConstr(
            quicksum(x3[i] for i in idxs) <= 1, name=f"NS_{group}_mutual_exclusivity"
        )

    stage3_model.addConstr(
        quicksum(x3[i] * revenue[i] for i in ns_strict_indices) == -min_ns_spending,
        name="ExactNSSpending",
    )

    # Additional constraints: Fix GDP and jobs to optimal values
    stage3_model.addConstr(quicksum(x3[i] * gdp[i] for i in indices) == gdp_star, name="GDPMatch")
    stage3_model.addConstr(
        quicksum(x3[i] * jobs[i] for i in indices) == jobs_star, name="JobsMatch"
    )

    # Solve Stage 3
    try:
        stage3_model.optimize()
    except GurobiError:
        if verbose:
            logger.exception("Stage 3 optimization failed")
        raise

    if stage3_model.status != GRB.OPTIMAL:
        error_msg = f"Stage 3 optimization failed with status {stage3_model.status}"
        if verbose:
            logger.error(error_msg)
        raise ValueError(f"{error_msg}. This should not happen if Stages 1 and 2 succeeded.")

    # Extract final solution from Stage 3
    selected_indices = [i for i in indices if x3[i].X > BINARY_THRESHOLD]
    selected_df = df.iloc[selected_indices].copy()

    # Calculate KPI values from final solution
    kpi_dict: dict[str, float] = {
        "GDP": gdp_star,
        "Jobs": jobs_star,
        "Revenue": stage3_model.ObjVal,
        "Capital": float(sum(selected_df[COLUMNS["capital"]])),
        "Wage": float(sum(selected_df[COLUMNS["wage"]])),
        "P20": float(sum(selected_df[COLUMNS["p20"]])),
        "P40-60": float(sum(selected_df[COLUMNS["p40_60"]])),
        "P80-100": float(sum(selected_df[COLUMNS["p80_100"]])),
        "P99": float(sum(selected_df[COLUMNS["p99"]])),
    }

    return selected_df, gdp_star, jobs_star, stage3_model.ObjVal, kpi_dict


def run_single_optimization(spending_level: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Run optimization for a single defense spending level.

    Args:
        spending_level: Defense spending requirement in billions

    Returns:
        tuple: (result_df, kpi_dict) for aggregation in run_full_range

    Raises:
        ValidationError: If inputs are invalid
        GurobiError: If optimization fails
    """
    # Validate spending level
    validate_spending_level(spending_level)

    # Load and clean data
    df, ns_groups = load_policy_data()
    ns_strict_indices = get_ns_strict_indices(df)

    # Run optimization with specified spending level
    result_df, gdp_impact, jobs_impact, revenue_impact, kpi_dict = optimize_policy_selection(
        df, ns_groups, ns_strict_indices, min_ns_spending=spending_level
    )

    # Display results
    display_results(result_df, gdp_impact, revenue_impact)

    logger.info("\n[Three-Stage Optimization Results]")
    logger.info(f"  Stage 1 - Optimal GDP: {gdp_impact * 100:.4f}%")
    logger.info(f"  Stage 2 - Optimal Jobs: {jobs_impact:,.0f}")
    logger.info(f"  Stage 3 - Optimal Revenue: ${revenue_impact:,.2f}B")

    # Display distributional outcomes
    logger.info("\n[Progressive Distribution Verification]")
    p20_val = kpi_dict["P20"] * 100
    p40_val = kpi_dict["P40-60"] * 100
    p80_val = kpi_dict["P80-100"] * 100
    p99_val = kpi_dict["P99"] * 100

    logger.info(f"  P20 (Bottom 20%):    {p20_val:>+8.4f}%")
    logger.info(f"  P40-60 (Middle):     {p40_val:>+8.4f}%")
    logger.info(f"  P80-100 (Top 20%):   {p80_val:>+8.4f}%")
    logger.info(f"  P99 (Top 1%):        {p99_val:>+8.4f}%")

    logger.info("\n  [Progressive Constraints Check]")
    logger.info(f"  P20 > P80-100: {p20_val:.4f}% > {p80_val:.4f}% = {p20_val > p80_val}")
    logger.info(f"  P20 > P99:     {p20_val:.4f}% > {p99_val:.4f}% = {p20_val > p99_val}")
    logger.info(f"  P40 > P80-100: {p40_val:.4f}% > {p80_val:.4f}% = {p40_val > p80_val}")
    logger.info(f"  P40 > P99:     {p40_val:.4f}% > {p99_val:.4f}% = {p40_val > p99_val}")

    # Save to CSV with spending level in filename
    output_file = f"outputs/senate/policy_portfolio_{spending_level}.csv"
    try:
        result_df.to_csv(output_file, index=False)
        logger.info(f"\n[OK] Results saved to '{output_file}'")
    except Exception:
        logger.exception("Failed to save results")
        raise

    return result_df, kpi_dict


def run_full_range() -> None:  # noqa: PLR0915
    """Run optimization for the full range of defense spending levels and generate visualization."""
    # Ensure output directory exists
    output_dir = Path("outputs/senate")
    try:
        validate_output_directory(output_dir)
    except ValidationError:
        logger.exception("Cannot create output directory")
        raise

    # Defense spending levels from config (same as max_gdp_defense.py)
    spending_levels = list(
        range(SPENDING_RANGE["min"], SPENDING_RANGE["max"], SPENDING_RANGE["step"])
    )

    logger.info(
        "Generating policy portfolio optimization results for full defense spending range..."
    )
    logger.info(
        f"Range: ${SPENDING_RANGE['min']:,}B to ${SPENDING_RANGE['max'] - SPENDING_RANGE['step']:,}B"
    )
    logger.info(f"Increment: ${SPENDING_RANGE['step']:,}B")
    logger.info("=" * 70)

    # Load policy data once to get all policy names
    df, _ = load_policy_data()
    all_policy_names = df[COLUMNS["option"]].tolist()

    # Initialize data structures for summary outputs
    policy_decisions: dict[int, dict[str, int]] = {}  # {spending_level: {policy_name: 0 or 1}}
    kpi_summary: dict[int, dict[str, float]] = {}  # {spending_level: {kpi_name: value}}

    successful_runs: list[int] = []
    failed_runs: list[int] = []

    for level in spending_levels:
        logger.info(f"\nRunning optimization for ${level:,}B defense spending...")
        try:
            # Run optimization directly
            result_df, kpi_dict = run_single_optimization(level)

            successful_runs.append(level)
            logger.info(f"[OK] Successfully generated policy_portfolio_{level}.csv")

            # Track policy decisions
            selected_policies = set(result_df[COLUMNS["option"]].tolist())
            policy_decisions[level] = {
                policy: 1 if policy in selected_policies else 0 for policy in all_policy_names
            }

            # Track KPI values
            kpi_summary[level] = kpi_dict

        except ValidationError:
            failed_runs.append(level)
            logger.exception(f"[FAILED] Validation failed for ${level:,}B")
        except GurobiError:
            failed_runs.append(level)
            logger.exception(f"[FAILED] Optimization failed for ${level:,}B")
        except Exception:
            failed_runs.append(level)
            logger.exception(f"[FAILED] Unexpected error for ${level:,}B")

    logger.info("\n" + "=" * 70)
    logger.info(f"Completed {len(successful_runs)}/{len(spending_levels)} optimization runs")

    if failed_runs:
        logger.warning(f"Failed runs: {failed_runs}")

    # Generate summary outputs if we have results
    if successful_runs:
        logger.info("\n" + "=" * 70)
        logger.info("Generating summary outputs...")

        try:
            # Create policy decision matrix (policies as rows, spending levels as columns)
            policy_matrix_df = pd.DataFrame(policy_decisions)
            policy_matrix_df.index.name = "Policy"
            policy_matrix_file = "outputs/senate/policy_decisions_matrix.csv"
            policy_matrix_df.to_csv(policy_matrix_file)
            logger.info(f"[OK] Policy decision matrix saved to '{policy_matrix_file}'")
            logger.info("  Format: Policies as rows, defense spending levels as columns")

            # Create KPI summary matrix
            kpi_matrix_df = pd.DataFrame(kpi_summary).T
            kpi_matrix_df.index.name = "Defense_Spending_B"
            kpi_summary_file = "outputs/senate/economic_effects_summary.csv"
            kpi_matrix_df.to_csv(kpi_summary_file)
            logger.info(f"[OK] Economic effects summary saved to '{kpi_summary_file}'")
        except Exception:
            logger.exception("Failed to save summary files")

        # Run visualizations if we have results
        logger.info("\n" + "=" * 70)
        logger.info("Generating visualizations...")

        # Note: Visualization scripts will need to be updated to read from outputs/senate/

        # 1. Defense spending analysis visualization
        try:
            result = subprocess.run(
                [sys.executable, "visualize_defense_spending.py", "--senate"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(result.stdout)  # noqa: T201
            logger.info("[OK] Defense spending visualization complete!")
        except subprocess.CalledProcessError as e:
            logger.exception("[FAILED] Defense spending visualization failed")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")  # noqa: TRY400
        except FileNotFoundError:
            logger.warning("[WARNING] visualize_defense_spending.py not found, skipping")

        # 2. Policy selection heatmap visualization
        try:
            result = subprocess.run(
                [sys.executable, "visualize_policy_selection.py", "--senate"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(result.stdout)  # noqa: T201
            logger.info("[OK] Policy selection visualization complete!")
        except subprocess.CalledProcessError as e:
            logger.exception("[FAILED] Policy selection visualization failed")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")  # noqa: TRY400
        except FileNotFoundError:
            logger.warning("[WARNING] visualize_policy_selection.py not found, skipping")

        logger.info("\n" + "=" * 70)
        logger.info("All visualizations complete!")


def main() -> None:
    """Main execution function with comprehensive error handling."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimize policy portfolio with progressive distribution and required/prohibited policies.",
        epilog="Run without arguments to generate full range of scenarios.",
    )
    parser.add_argument(
        "--spending", type=int, help="Run single optimization with specific NS spending in billions"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly run full range of spending levels",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        set_global_level(LogLevel.DEBUG)

    try:
        # Determine mode of operation
        if args.spending is not None:
            # Single optimization run
            logger.info(f"Starting single optimization for ${args.spending:,}B defense spending")
            run_single_optimization(args.spending)
        else:
            # Default: Run full range
            logger.info("Starting full range optimization")
            run_full_range()

    except ValidationError:
        logger.exception("Validation error")
        sys.exit(1)
    except GurobiError:
        logger.exception(
            "Gurobi optimization error. Please check your Gurobi license and model formulation."
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("Unexpected error")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
