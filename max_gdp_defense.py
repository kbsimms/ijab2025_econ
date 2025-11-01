"""
Policy Optimization Script with Equity, Policy, and National Security Constraints

WHAT THIS SCRIPT DOES:
This script finds the best combination of economic policies to maximize GDP growth
while ensuring:
- The government doesn't increase the deficit
- Economic outcomes are positive (jobs, wages, capital all improve)
- Lower and middle-income families benefit at least as much as wealthy families
- National defense gets adequate funding
- Policy choices are coherent (no contradictory policies selected together)

Think of it as answering: "Which policies grow the economy the most while being
fair to working families, maintaining strong defense, and not adding to the debt?"

HOW IT WORKS:
Uses advanced mathematical optimization to evaluate billions of policy combinations
and find the absolute best one that meets all requirements. Can analyze:
- Single defense spending level (e.g., "$3,000B for defense")
- Full range of defense levels (-$4,000B to +$6,000B) to show trade-offs

CONSTRAINTS (Requirements every solution must meet):

1. FISCAL: Revenue neutrality - can't increase the federal deficit
   
2. ECONOMIC: Must improve the economy across the board
   - More jobs created
   - Higher wages
   - More capital investment
   
3. EQUITY: Progressive benefits - working families come first
   - Bottom 20% (P20) must benefit at least as much as top 20% (P80-100)
   - Middle class (P40-60) must benefit at least as much as wealthy (P80-100, P99)
   - Everyone's after-tax income must increase (no one worse off)
   
4. POLICY COHERENCE: Can't select contradictory policies
   - 15 mutually exclusive groups (e.g., can't have two different corporate tax rates)
   - Certain policies explicitly forbidden (e.g., new VAT taxes)
   
5. NATIONAL SECURITY: Adequate defense funding
   - Can specify exact defense spending requirement (e.g., $3,000B)
   - Only one option per defense category (e.g., can't both increase AND decrease)

OPTIMIZATION APPROACH:
Two-stage process ensures the very best solution:
- Stage 1: Find maximum possible GDP growth meeting all constraints
- Stage 2: Among all max-GDP solutions, pick one with highest revenue surplus

USAGE EXAMPLES:
    # Run full analysis across all defense spending levels
    python max_gdp_defense.py
    
    # Run single optimization with $3,000B defense spending
    python max_gdp_defense.py --spending 3000
    
    # Run full range (same as default, but explicit)
    python max_gdp_defense.py --all

OUTPUTS:
- CSV files with selected policies for each spending level
- Comprehensive visualizations showing economic trade-offs
- Summary files with policy decisions and economic effects
"""

import argparse
import subprocess
import sys
from typing import Tuple
from pathlib import Path
import pandas as pd
from gurobipy import Model, GRB, quicksum, GurobiError

from config import (
    COLUMNS,
    SUPPRESS_GUROBI_OUTPUT,
    EPSILON,
    DISTRIBUTIONAL_TOLERANCE,
    DEFENSE_SPENDING,
    SPENDING_RANGE,
    EXCLUDED_POLICIES
)
from utils import load_policy_data, get_ns_strict_indices, display_results
from logger import get_logger, LogLevel
from validation import (
    validate_optimization_inputs,
    validate_spending_level,
    validate_output_directory,
    ValidationError
)
from optimizer_utils import (
    get_policy_indices_by_codes,
    add_all_constraints
)

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)


def define_policy_groups(df: pd.DataFrame) -> dict:
    """
    Define mutually exclusive policy groups.
    
    Returns:
        Dictionary mapping group names to lists of policy indices
    """
    policy_groups = {}
    
    # 1. Corporate Tax Rate/Structure
    policy_groups['corporate_tax'] = get_policy_indices_by_codes(df, ['11', '36', '68'])
    
    # 2. Gas Tax Increases
    policy_groups['gas_tax'] = get_policy_indices_by_codes(df, ['47', '48'])
    
    # 3. Estate Tax
    policy_groups['estate_tax'] = get_policy_indices_by_codes(df, ['12', '44', '46', '69'])
    
    # 4. Child Tax Credit - Refundability
    policy_groups['ctc_refundability'] = get_policy_indices_by_codes(df, ['53', '54'])
    
    # 5. Social Security Payroll Tax Cap
    policy_groups['ss_payroll_cap'] = get_policy_indices_by_codes(df, ['34', '35'])
    
    # 6. Payroll Tax Rate Changes
    policy_groups['payroll_rate'] = get_policy_indices_by_codes(df, ['4', '33'])
    
    # 7. EITC/CDCTC Reforms
    policy_groups['eitc_reforms'] = get_policy_indices_by_codes(df, ['21', '51', '52', '55', 'S15'])
    
    # 8. Individual Income Tax Structure
    policy_groups['individual_tax_structure'] = get_policy_indices_by_codes(df, ['1', '3', '14', '59'])
    
    # 9. Child Tax Credit - Comprehensive
    policy_groups['ctc_comprehensive'] = get_policy_indices_by_codes(df, ['19', '20', '55', 'S13'])
    
    # 10. Section 199A Deduction
    policy_groups['section_199a'] = get_policy_indices_by_codes(df, ['10', '38'])
    
    # 11. Home Mortgage Interest Deduction
    policy_groups['mortgage_deduction'] = get_policy_indices_by_codes(df, ['23', '24'])
    
    # 12. Charitable Deduction
    policy_groups['charitable_deduction'] = get_policy_indices_by_codes(df, ['25', '58'])
    
    # 13. Capital Gains Tax Rate
    policy_groups['capital_gains'] = get_policy_indices_by_codes(df, ['5', '29', '30'])
    
    # 14. Depreciation/Expensing
    policy_groups['depreciation'] = get_policy_indices_by_codes(df, ['7', '40', '65'])
    
    # 15. Value Added Tax (VAT)
    policy_groups['vat'] = get_policy_indices_by_codes(df, ['43', '68'])
    
    # Remove empty groups
    policy_groups = {k: v for k, v in policy_groups.items() if len(v) > 0}
    
    return policy_groups


def optimize_policy_selection(
    df: pd.DataFrame,
    ns_groups: dict,
    ns_strict_indices: list,
    min_ns_spending: int = DEFENSE_SPENDING["baseline"],
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, float, dict]:
    """
    Two-stage optimization with equity, policy, and national security constraints.
    
    Stage 1: Maximize GDP subject to all constraints
        - Fiscal: Revenue neutrality (sum of dynamic revenue >= 0)
        - Economic: Non-negative capital stock, jobs, wage rate
        - Equity: Progressive distribution (P20, P40-60 >= P80-100, P99)
        - Income: All income groups must have non-negative after-tax income (everyone better off)
        - Policy exclusions: Policies {37, 43, 49, 68} cannot be selected
        - Policy mutual exclusivity: At most one policy per competing group (15 groups)
        - Special policy constraints: E.g., VAT replacement excludes corporate surtax
        - NS mutual exclusivity: At most one policy per NS group
        - NS spending target: Exactly min_ns_spending on NS1-NS7 policies
        
    Stage 2: Maximize revenue while maintaining optimal GDP
        - Same constraints as Stage 1
        - Additionally constrains GDP to equal the optimal value from Stage 1
        - Breaks ties by maximizing revenue surplus
    
    Equity Constraints Explained:
        The equity constraints ensure progressive policy impacts:
        1. P20 and P40-60 must individually benefit at least as much as P80-100 and P99
        2. All income groups (P20, P40-60, P80-100, P99) must have >= 0 after-tax income effects
    
    Policy Mutual Exclusivity Groups:
        1. Corporate Tax Rate/Structure: {11, 36, 68}
        2. Gas Tax Increases: {47, 48}
        3. Estate Tax: {12, 44, 46, 69}
        4. Child Tax Credit - Refundability: {53, 54}
        5. Social Security Payroll Tax Cap: {34, 35}
        6. Payroll Tax Rate Changes: {4, 33}
        7. EITC Reforms: {21, 51, 52, 55, S15}
        8. Individual Income Tax Structure: {1, 3, 14, 59}
        9. Child Tax Credit - Comprehensive: {19, 20, 55, S13}
        10. Section 199A Deduction: {10, 38}
        11. Home Mortgage Interest Deduction: {23, 24}
        12. Charitable Deduction: {25, 58}
        13. Capital Gains Tax Rate: {5, 29, 30}
        14. Depreciation/Expensing: {7, 40, 65}
        15. Value Added Tax (VAT): {43, 68}
        
        Excluded Policies (forced to 0):
        - 37: Corporate Surtax of 5%
        - 43: Enact a 5% VAT
        - 49: Reinstate the Cadillac Tax
        - 68: Replace CIT with 5% VAT
        
        Special: If 68 (Replace CIT with VAT), then not 37 (Corporate Surtax)
        Note: Both 68 and 37 are already in excluded list, so this constraint is redundant
        
    Args:
        df: DataFrame containing policy options and their impacts
        ns_groups: Dict mapping NS group names to lists of policy indices
        ns_strict_indices: List of indices for NS1-NS7 policies
        min_ns_spending: Minimum NS spending in billions (default: 3000)
        verbose: If True, prints progress messages
        
    Returns:
        tuple: (selected_df, gdp_impact, revenue_impact, kpi_dict)
            - selected_df: DataFrame of selected policies
            - gdp_impact: Total GDP impact achieved
            - revenue_impact: Total revenue impact achieved
            - kpi_dict: Dictionary of all KPI values
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
    
    # Validate inputs before optimization
    try:
        validate_optimization_inputs(df, ns_groups, ns_strict_indices, min_ns_spending)
    except ValidationError as e:
        if verbose:
            logger.error(f"Input validation failed: {e}")
        raise
    
    # === Stage 1: Maximize GDP ===
    if verbose:
        logger.info(f"Running optimization (Stage 1: Maximize GDP)...")
        logger.info(f"  NS spending requirement: ${min_ns_spending:,}B")
        
        # Get policy groups for display
        policy_groups_display = define_policy_groups(df)
        if policy_groups_display:
            logger.info(f"  Policy mutual exclusivity groups: {len(policy_groups_display)}")
    
    try:
        stage1_model = Model("Stage1_MaximizeGDP")
    except GurobiError as e:
        if verbose:
            logger.error(f"Failed to create Gurobi model: {e}")
        raise
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
    # Use centralized constraint functions to eliminate code duplication
    policy_groups = define_policy_groups(df)
    
    # Add all standard constraints using utility function
    # This replaces ~100 lines of duplicate constraint code
    add_all_constraints(
        stage1_model, x, df, ns_groups, policy_groups,
        ns_strict_indices, min_ns_spending,
        logger=logger if verbose else None
    )
    
    # Solve Stage 1
    try:
        stage1_model.optimize()
    except GurobiError as e:
        if verbose:
            logger.error(f"Stage 1 optimization failed: {e}")
        raise
    
    # Check if model found a feasible solution
    if stage1_model.status != GRB.OPTIMAL:
        error_msg = f"Stage 1 optimization failed with status {stage1_model.status}"
        if verbose:
            logger.error(error_msg)
            if stage1_model.status == GRB.INFEASIBLE:
                logger.error("Model is infeasible. Possible causes:")
                logger.error(f"  - NS spending requirement too restrictive: ${min_ns_spending:,}B")
                logger.error("  - Equity constraints cannot be satisfied")
                logger.error("  - Policy exclusions create conflicts")
            elif stage1_model.status == GRB.UNBOUNDED:
                logger.error("Model is unbounded")
        raise ValueError(
            f"{error_msg}. "
            f"The model may be infeasible - try reducing the NS spending requirement. "
            f"Current requirement: ${min_ns_spending:,}B"
        )
    
    gdp_star = stage1_model.ObjVal
    if verbose:
        logger.debug(f"Stage 1 optimal GDP: {gdp_star * 100:.4f}%")
    
    # === Stage 2: Maximize Revenue under optimal GDP ===
    if verbose:
        logger.info("Running optimization (Stage 2: Maximize Revenue)...")
    
    try:
        stage2_model = Model("Stage2_MaximizeRevenue")
    except GurobiError as e:
        if verbose:
            logger.error(f"Failed to create Stage 2 model: {e}")
        raise
    if SUPPRESS_GUROBI_OUTPUT:
        stage2_model.setParam("OutputFlag", 0)
    
    # New decision variables for Stage 2
    x2 = stage2_model.addVars(indices, vtype=GRB.BINARY, name="x")
    
    # Objective: Maximize dynamic revenue (break ties from Stage 1)
    stage2_model.setObjective(
        quicksum(x2[i] * revenue[i] for i in indices),
        GRB.MAXIMIZE
    )
    
    # Add all constraints (same as Stage 1, but with x2 variables)
    add_all_constraints(
        stage2_model, x2, df, ns_groups, policy_groups,
        ns_strict_indices, min_ns_spending,
        logger=logger if verbose else None
    )
    
    # Additional constraint: Fix GDP to the optimal value from Stage 1
    # This ensures we don't sacrifice GDP to gain more revenue
    stage2_model.addConstr(
        quicksum(x2[i] * gdp[i] for i in indices) == gdp_star,
        name="GDPMatch"
    )
    
    # Solve Stage 2
    try:
        stage2_model.optimize()
    except GurobiError as e:
        if verbose:
            logger.error(f"Stage 2 optimization failed: {e}")
        raise
    
    # Check if model found a feasible solution
    if stage2_model.status != GRB.OPTIMAL:
        error_msg = f"Stage 2 optimization failed with status {stage2_model.status}"
        if verbose:
            logger.error(error_msg)
        raise ValueError(
            f"{error_msg}. "
            "This should not happen if Stage 1 succeeded."
        )
    
    # Extract solution: policies where x2[i] > 0.5 are selected
    # Using 0.5 threshold handles numerical precision issues in binary variables
    selected_indices = [i for i in indices if x2[i].X > 0.5]
    selected_df = df.iloc[selected_indices].copy()
    
    # Calculate KPI values
    kpi_dict = {
        'GDP': gdp_star,
        'Revenue': stage2_model.ObjVal,
        'Capital': sum(selected_df[COLUMNS["capital"]]),
        'Jobs': sum(selected_df[COLUMNS["jobs"]]),
        'Wage': sum(selected_df[COLUMNS["wage"]]),
        'P20': sum(selected_df[COLUMNS["p20"]]),
        'P40-60': sum(selected_df[COLUMNS["p40_60"]]),
        'P80-100': sum(selected_df[COLUMNS["p80_100"]]),
        'P99': sum(selected_df[COLUMNS["p99"]])
    }
    
    return selected_df, gdp_star, stage2_model.ObjVal, kpi_dict


def run_single_optimization(spending_level: int) -> Tuple[pd.DataFrame, dict]:
    """
    Run optimization for a single spending level.
    
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
    result_df, gdp_impact, revenue_impact, kpi_dict = optimize_policy_selection(
        df, ns_groups, ns_strict_indices, min_ns_spending=spending_level
    )
    
    # Display results
    display_results(result_df, gdp_impact, revenue_impact)
    
    # Save to CSV with spending level in filename
    output_file = f"outputs/defense/max_gdp_defense{spending_level}.csv"
    try:
        result_df.to_csv(output_file, index=False)
        logger.info(f"[OK] Results saved to '{output_file}'")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise
    
    return result_df, kpi_dict


def run_full_range() -> None:
    """Run optimization for the full range of defense spending levels and generate visualization."""
    # Ensure output directory exists
    output_dir = Path("outputs/defense")
    try:
        validate_output_directory(output_dir)
    except ValidationError as e:
        logger.error(f"Cannot create output directory: {e}")
        raise
    
    # Defense spending levels from config
    spending_levels = list(range(
        SPENDING_RANGE["min"],
        SPENDING_RANGE["max"],
        SPENDING_RANGE["step"]
    ))
    
    logger.info("Generating optimization results for full defense spending range...")
    logger.info(f"Range: ${SPENDING_RANGE['min']:,}B to ${SPENDING_RANGE['max']-SPENDING_RANGE['step']:,}B")
    logger.info(f"Increment: ${SPENDING_RANGE['step']:,}B")
    logger.info("=" * 70)
    
    # Load policy data once to get all policy names
    df, ns_groups = load_policy_data()
    ns_strict_indices = get_ns_strict_indices(df)
    all_policy_names = df[COLUMNS["option"]].tolist()
    
    # Initialize data structures for summary outputs
    policy_decisions = {}  # {spending_level: {policy_name: 0 or 1}}
    kpi_summary = {}  # {spending_level: {kpi_name: value}}
    
    successful_runs = []
    failed_runs = []
    
    for level in spending_levels:
        logger.info(f"\nRunning optimization for ${level:,}B defense spending...")
        try:
            # Validate spending level
            validate_spending_level(level)
            
            # Run optimization directly (not via subprocess)
            result_df, kpi_dict = run_single_optimization(level)
            
            successful_runs.append(level)
            logger.info(f"[OK] Successfully generated max_gdp_defense{level}.csv")
            
            # Track policy decisions
            selected_policies = set(result_df[COLUMNS["option"]].tolist())
            policy_decisions[level] = {
                policy: 1 if policy in selected_policies else 0
                for policy in all_policy_names
            }
            
            # Track KPI values
            kpi_summary[level] = kpi_dict
            
        except ValidationError as e:
            failed_runs.append(level)
            logger.error(f"[FAILED] Validation failed for ${level:,}B: {e}")
        except GurobiError as e:
            failed_runs.append(level)
            logger.error(f"[FAILED] Optimization failed for ${level:,}B: {e}")
        except Exception as e:
            failed_runs.append(level)
            logger.error(f"[FAILED] Unexpected error for ${level:,}B: {e}")
    
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
            policy_matrix_df.index.name = 'Policy'
            policy_matrix_file = "outputs/defense/policy_decisions_matrix.csv"
            policy_matrix_df.to_csv(policy_matrix_file)
            logger.info(f"[OK] Policy decision matrix saved to '{policy_matrix_file}'")
            logger.info(f"  Format: Policies as rows, defense spending levels as columns")
            
            # Create KPI summary matrix
            kpi_matrix_df = pd.DataFrame(kpi_summary).T
            kpi_matrix_df.index.name = 'Defense_Spending_B'
            kpi_summary_file = "outputs/defense/economic_effects_summary.csv"
            kpi_matrix_df.to_csv(kpi_summary_file)
            logger.info(f"[OK] Economic effects summary saved to '{kpi_summary_file}'")
        except Exception as e:
            logger.error(f"Failed to save summary files: {e}")
        
        # Run visualization if we have results
        logger.info("\n" + "=" * 70)
        logger.info("Generating visualization...")
        try:
            result = subprocess.run(
                [sys.executable, "visualize_defense_spending.py"],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)  # Print visualization output directly
            logger.info("[OK] Visualization complete!")
        except subprocess.CalledProcessError as e:
            logger.error("[FAILED] Visualization failed:")
            logger.error(e.stderr)
        except FileNotFoundError:
            logger.warning("[WARNING] visualize_defense_spending.py not found, skipping visualization")


def main() -> None:
    """Main execution function with comprehensive error handling."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Optimize policy selection with national security and equity constraints.',
        epilog='Run without arguments to generate full range of scenarios and visualization.'
    )
    parser.add_argument(
        '--spending',
        type=int,
        help='Run single optimization with specific NS spending in billions'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Explicitly run full range of spending levels and generate visualization'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug-level logging'
    )
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        from logger import set_global_level
        set_global_level(LogLevel.DEBUG)
    
    try:
        # Determine mode of operation
        if args.spending is not None:
            # Single optimization run
            logger.info(f"Starting single optimization for ${args.spending:,}B defense spending")
            run_single_optimization(args.spending)
        else:
            # Default: Run full range + visualization
            logger.info("Starting full range optimization")
            run_full_range()
            
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except GurobiError as e:
        logger.error(f"Gurobi optimization error: {e}")
        logger.error("Please check your Gurobi license and model formulation.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()