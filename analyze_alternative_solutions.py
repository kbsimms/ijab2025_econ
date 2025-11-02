"""
Alternative Solutions Analysis for Defense Spending Optimization

WHAT THIS SCRIPT DOES:
For each defense spending level, this script:
1. Runs Stage 1 optimization to find maximum GDP
2. Enumerates ALL alternative policy combinations that achieve the same maximum GDP
3. Analyzes variation in secondary objectives (jobs, revenue, equity, etc.)
4. Generates a comprehensive report to inform Stage 2 objective selection

This helps answer: "How many different ways can we achieve maximum GDP at each
spending level, and how much do they differ in jobs, revenue, and equity?"

USAGE:
    python analyze_alternative_solutions.py

OUTPUT:
- Console report with detailed analysis for each spending level
- CSV file: outputs/defense/alternative_solutions_analysis.csv
"""

from pathlib import Path
from typing import Any

from gurobipy import GRB, GurobiError, Model, quicksum
import pandas as pd

from config import COLUMNS, SPENDING_RANGE, SUPPRESS_GUROBI_OUTPUT
from logger import LogLevel, get_logger
from optimizer_utils import add_all_constraints, get_policy_indices_by_codes
from utils import get_ns_strict_indices, load_policy_data
from validation import validate_spending_level

# Initialize logger
logger = get_logger(__name__, level=LogLevel.INFO)

# Binary variable decision threshold
BINARY_THRESHOLD = 0.5

# Maximum number of solutions to enumerate per spending level
MAX_SOLUTIONS_PER_LEVEL = 100


def define_policy_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """Define mutually exclusive policy groups (copied from max_gdp_defense.py)."""
    policy_groups: dict[str, list[int]] = {}

    policy_groups["corporate_tax"] = get_policy_indices_by_codes(df, ["11", "36", "68"])
    policy_groups["gas_tax"] = get_policy_indices_by_codes(df, ["47", "48"])
    policy_groups["estate_tax"] = get_policy_indices_by_codes(df, ["12", "44", "46", "69"])
    policy_groups["ctc_refundability"] = get_policy_indices_by_codes(df, ["53", "54"])
    policy_groups["ss_payroll_cap"] = get_policy_indices_by_codes(df, ["34", "35"])
    policy_groups["payroll_rate"] = get_policy_indices_by_codes(df, ["4", "33"])
    policy_groups["eitc_reforms"] = get_policy_indices_by_codes(df, ["21", "51", "52", "55", "S15"])
    policy_groups["individual_tax_structure"] = get_policy_indices_by_codes(
        df, ["1", "3", "14", "59"]
    )
    policy_groups["ctc_comprehensive"] = get_policy_indices_by_codes(df, ["19", "20", "55", "S13"])
    policy_groups["section_199a"] = get_policy_indices_by_codes(df, ["10", "38"])
    policy_groups["mortgage_deduction"] = get_policy_indices_by_codes(df, ["23", "24"])
    policy_groups["charitable_deduction"] = get_policy_indices_by_codes(df, ["25", "58"])
    policy_groups["capital_gains"] = get_policy_indices_by_codes(df, ["5", "29", "30"])
    policy_groups["depreciation"] = get_policy_indices_by_codes(df, ["7", "40", "65"])
    policy_groups["vat"] = get_policy_indices_by_codes(df, ["43", "68"])

    return {k: v for k, v in policy_groups.items() if len(v) > 0}


def analyze_alternatives_for_spending_level(
    df: pd.DataFrame,
    ns_groups: dict[str, list[int]],
    ns_strict_indices: list[int],
    min_ns_spending: int,
) -> dict[str, Any]:
    """
    Enumerate alternative optimal solutions for a specific spending level.

    Returns:
        Dictionary with analysis results including solution count and metric variations
    """
    n = len(df)
    indices = range(n)

    # Extract data arrays
    gdp = df[COLUMNS["gdp"]].values
    revenue = df[COLUMNS["dynamic_revenue"]].values
    jobs = df[COLUMNS["jobs"]].values
    capital = df[COLUMNS["capital"]].values
    wage = df[COLUMNS["wage"]].values
    p20_arr = df[COLUMNS["p20"]].values
    p40_arr = df[COLUMNS["p40_60"]].values
    p80_arr = df[COLUMNS["p80_100"]].values
    p99_arr = df[COLUMNS["p99"]].values

    # === Stage 1: Find optimal GDP ===
    try:
        model = Model("FindOptimalGDP")
    except GurobiError:
        logger.exception("Failed to create Gurobi model")
        raise

    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam("OutputFlag", 0)

    # Configure solution pool to find multiple optimal solutions
    model.setParam("PoolSearchMode", 2)  # Systematic search for best solutions
    model.setParam("PoolSolutions", MAX_SOLUTIONS_PER_LEVEL)  # Store up to 100 solutions
    model.setParam("PoolGap", 0.0)  # Only accept solutions equal to optimal

    # Decision variables
    x = model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Objective: Maximize GDP
    model.setObjective(quicksum(x[i] * gdp[i] for i in indices), GRB.MAXIMIZE)

    # Add all constraints
    policy_groups = define_policy_groups(df)
    add_all_constraints(model, x, df, ns_groups, policy_groups, ns_strict_indices, min_ns_spending)

    # Solve
    try:
        model.optimize()
    except GurobiError:
        logger.exception("Optimization failed")
        raise

    if model.status != GRB.OPTIMAL:
        raise ValueError(f"Model is not optimal. Status: {model.status}")

    optimal_gdp = model.ObjVal
    solution_count = model.SolCount

    logger.info(f"  Found {solution_count} alternative optimal solutions")
    logger.info(f"  Optimal GDP: {optimal_gdp * 100:.4f}%")

    # Analyze variation across all solutions in the pool
    solutions_data = []
    for sol_idx in range(solution_count):
        model.setParam("SolutionNumber", sol_idx)

        # Extract this solution
        selected = [i for i in indices if x[i].Xn > BINARY_THRESHOLD]

        # Calculate metrics for this solution
        sol_revenue = sum(revenue[i] for i in selected)
        sol_jobs = sum(jobs[i] for i in selected)
        sol_capital = sum(capital[i] for i in selected)
        sol_wage = sum(wage[i] for i in selected)
        sol_p20 = sum(p20_arr[i] for i in selected)
        sol_p40 = sum(p40_arr[i] for i in selected)
        sol_p80 = sum(p80_arr[i] for i in selected)
        sol_p99 = sum(p99_arr[i] for i in selected)

        solutions_data.append(
            {
                "solution": sol_idx + 1,
                "revenue": sol_revenue,
                "jobs": sol_jobs,
                "capital": sol_capital,
                "wage": sol_wage,
                "p20": sol_p20,
                "p40_60": sol_p40,
                "p80_100": sol_p80,
                "p99": sol_p99,
                "n_policies": len(selected),
            }
        )

    # Calculate variation statistics
    solutions_df = pd.DataFrame(solutions_data)

    analysis = {
        "spending_level": min_ns_spending,
        "optimal_gdp": optimal_gdp,
        "solution_count": solution_count,
        "revenue_min": solutions_df["revenue"].min(),
        "revenue_max": solutions_df["revenue"].max(),
        "revenue_range": solutions_df["revenue"].max() - solutions_df["revenue"].min(),
        "jobs_min": solutions_df["jobs"].min(),
        "jobs_max": solutions_df["jobs"].max(),
        "jobs_range": solutions_df["jobs"].max() - solutions_df["jobs"].min(),
        "capital_min": solutions_df["capital"].min(),
        "capital_max": solutions_df["capital"].max(),
        "wage_min": solutions_df["wage"].min(),
        "wage_max": solutions_df["wage"].max(),
        "p20_min": solutions_df["p20"].min(),
        "p20_max": solutions_df["p20"].max(),
        "equity_ratio_min": (solutions_df["p20"] / solutions_df["p99"]).min()
        if (solutions_df["p99"] > 0).all()
        else 0,
        "equity_ratio_max": (solutions_df["p20"] / solutions_df["p99"]).max()
        if (solutions_df["p99"] > 0).all()
        else 0,
    }

    return analysis  # noqa: RET504


def analyze_after_stage2(
    df: pd.DataFrame,
    ns_groups: dict[str, list[int]],
    ns_strict_indices: list[int],
    min_ns_spending: int,
    optimal_gdp: float,
    optimal_jobs: float,
) -> int:
    """
    Count alternative solutions after BOTH Stage 1 and Stage 2 constraints.

    This tells us: After fixing GDP=max and Jobs=max, how many solutions remain?
    If > 1, we might need a Stage 3 objective.

    Returns:
        Number of alternative solutions with both optimal GDP and optimal jobs
    """
    n = len(df)
    indices = range(n)

    # Extract data arrays
    gdp = df[COLUMNS["gdp"]].values
    jobs = df[COLUMNS["jobs"]].values

    try:
        model = Model("CountStage2Alternatives")
    except GurobiError:
        logger.exception("Failed to create model")
        raise

    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam("OutputFlag", 0)

    # Configure solution pool
    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", MAX_SOLUTIONS_PER_LEVEL)
    model.setParam("PoolGap", 0.0)

    # Decision variables
    x = model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Objective: Maximize any metric (doesn't matter since we're just counting)
    # Use revenue as arbitrary objective
    revenue = df[COLUMNS["dynamic_revenue"]].values
    model.setObjective(quicksum(x[i] * revenue[i] for i in indices), GRB.MAXIMIZE)

    # Add all standard constraints
    policy_groups = define_policy_groups(df)
    add_all_constraints(model, x, df, ns_groups, policy_groups, ns_strict_indices, min_ns_spending)

    # Additional constraints: Fix GDP and Jobs to their optimal values
    model.addConstr(quicksum(x[i] * gdp[i] for i in indices) == optimal_gdp, name="FixGDP")
    model.addConstr(quicksum(x[i] * jobs[i] for i in indices) == optimal_jobs, name="FixJobs")

    # Solve
    try:
        model.optimize()
    except GurobiError:
        logger.exception("Stage 2 enumeration failed")
        raise

    if model.status != GRB.OPTIMAL:
        return 0

    return model.SolCount


def analyze_after_stage3(
    df: pd.DataFrame,
    ns_groups: dict[str, list[int]],
    ns_strict_indices: list[int],
    min_ns_spending: int,
    optimal_gdp: float,
    optimal_jobs: float,
    optimal_revenue: float,
) -> tuple[int, dict[str, Any]]:
    """
    Count alternative solutions after Stage 3 (GDP, Jobs, AND Revenue all fixed).

    This tells us: After fixing GDP=max, Jobs=max, Revenue=max, how many solutions remain?
    If > 1, we need a Stage 4 objective.

    Returns:
        Tuple of (solution_count, variation_metrics)
    """
    n = len(df)
    indices = range(n)

    # Extract data arrays
    gdp = df[COLUMNS["gdp"]].values
    jobs = df[COLUMNS["jobs"]].values
    revenue = df[COLUMNS["dynamic_revenue"]].values
    capital = df[COLUMNS["capital"]].values
    wage = df[COLUMNS["wage"]].values
    p20_arr = df[COLUMNS["p20"]].values
    p99_arr = df[COLUMNS["p99"]].values

    try:
        model = Model("CountStage3Alternatives")
    except GurobiError:
        logger.exception("Failed to create model")
        raise

    if SUPPRESS_GUROBI_OUTPUT:
        model.setParam("OutputFlag", 0)

    # Configure solution pool
    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", MAX_SOLUTIONS_PER_LEVEL)
    model.setParam("PoolGap", 0.0)

    # Decision variables
    x = model.addVars(indices, vtype=GRB.BINARY, name="x")

    # Objective: Maximize any remaining metric (capital stock for variety)
    model.setObjective(quicksum(x[i] * capital[i] for i in indices), GRB.MAXIMIZE)

    # Add all standard constraints
    policy_groups = define_policy_groups(df)
    add_all_constraints(
        model, x, df, ns_groups, policy_groups, ns_strict_indices, min_ns_spending
    )

    # Fix all three optimized metrics
    model.addConstr(
        quicksum(x[i] * gdp[i] for i in indices) == optimal_gdp, name="FixGDP"
    )
    model.addConstr(
        quicksum(x[i] * jobs[i] for i in indices) == optimal_jobs, name="FixJobs"
    )
    model.addConstr(
        quicksum(x[i] * revenue[i] for i in indices) == optimal_revenue, name="FixRevenue"
    )

    # Solve
    try:
        model.optimize()
    except GurobiError:
        logger.exception("Stage 3 enumeration failed")
        raise

    if model.status != GRB.OPTIMAL:
        return 0, {}

    solution_count = model.SolCount

    # If multiple solutions exist, analyze variation in remaining metrics
    if solution_count > 1:
        solutions_data = []
        for sol_idx in range(solution_count):
            model.setParam("SolutionNumber", sol_idx)
            selected = [i for i in indices if x[i].Xn > BINARY_THRESHOLD]

            sol_capital = sum(capital[i] for i in selected)
            sol_wage = sum(wage[i] for i in selected)
            sol_p20 = sum(p20_arr[i] for i in selected)
            sol_p99 = sum(p99_arr[i] for i in selected)

            solutions_data.append({
                "capital": sol_capital,
                "wage": sol_wage,
                "p20": sol_p20,
                "p99": sol_p99,
                "equity_ratio": sol_p20 / sol_p99 if sol_p99 > 0 else 0,
            })

        solutions_df = pd.DataFrame(solutions_data)
        variation = {
            "capital_range": solutions_df["capital"].max() - solutions_df["capital"].min(),
            "wage_range": solutions_df["wage"].max() - solutions_df["wage"].min(),
            "equity_ratio_range": solutions_df["equity_ratio"].max()
            - solutions_df["equity_ratio"].min(),
        }
    else:
        variation = {}

    return solution_count, variation


def main() -> None:  # noqa: PLR0915
    """Main execution function."""
    logger.info("Starting Alternative Solutions Analysis...")
    logger.info("=" * 80)

    # Ensure output directory exists
    output_dir = Path("outputs/defense")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy data
    df, ns_groups = load_policy_data()
    ns_strict_indices = get_ns_strict_indices(df)

    # Spending levels to analyze
    spending_levels = list(
        range(SPENDING_RANGE["min"], SPENDING_RANGE["max"], SPENDING_RANGE["step"])
    )

    all_analyses = []

    for level in spending_levels:
        logger.info(f"\nAnalyzing ${level:,}B defense spending...")

        try:
            validate_spending_level(level)
            analysis = analyze_alternatives_for_spending_level(
                df, ns_groups, ns_strict_indices, level
            )

            # Count solutions after Stage 2
            if analysis["solution_count"] > 1:
                stage2_count = analyze_after_stage2(
                    df,
                    ns_groups,
                    ns_strict_indices,
                    level,
                    analysis["optimal_gdp"],
                    analysis["jobs_max"],
                )
                analysis["stage2_solution_count"] = stage2_count

                # Count solutions after Stage 3 (if Stage 2 has alternatives)
                if stage2_count > 1:
                    stage3_count, stage3_variation = analyze_after_stage3(
                        df,
                        ns_groups,
                        ns_strict_indices,
                        level,
                        analysis["optimal_gdp"],
                        analysis["jobs_max"],
                        analysis["revenue_max"],
                    )
                    analysis["stage3_solution_count"] = stage3_count
                    if stage3_variation:
                        analysis.update({f"stage3_{k}": v for k, v in stage3_variation.items()})
                else:
                    analysis["stage3_solution_count"] = 1
            else:
                analysis["stage2_solution_count"] = 1
                analysis["stage3_solution_count"] = 1

            all_analyses.append(analysis)

            # Display key findings
            logger.info(f"  Alternative solutions after Stage 1: {analysis['solution_count']}")
            logger.info(
                f"  Alternative solutions after Stage 2: {analysis['stage2_solution_count']}"
            )
            logger.info(
                f"  Alternative solutions after Stage 3: {analysis.get('stage3_solution_count', 'N/A')}"
            )
            if analysis["solution_count"] > 1:
                logger.info(
                    f"  Revenue range: ${analysis['revenue_min']:,.0f}B to ${analysis['revenue_max']:,.0f}B "
                    f"(Δ${analysis['revenue_range']:,.0f}B)"
                )
                logger.info(
                    f"  Jobs range: {analysis['jobs_min']:,.0f} to {analysis['jobs_max']:,.0f} "
                    f"(Δ{analysis['jobs_range']:,.0f})"
                )
                logger.info(
                    f"  Equity ratio range: {analysis['equity_ratio_min']:.2f} to {analysis['equity_ratio_max']:.2f}"
                )

        except Exception:
            logger.exception(f"Failed to analyze ${level:,}B")

    # Generate summary report
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: Alternative Solutions Count by Spending Level")
    logger.info("=" * 80)

    summary_df = pd.DataFrame(all_analyses)
    summary_file = output_dir / "alternative_solutions_analysis.csv"
    summary_df.to_csv(summary_file, index=False)

    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info(f"\n[OK] Full analysis saved to '{summary_file}'")

    # Key insights
    logger.info("\n" + "=" * 80)
    logger.info("KEY INSIGHTS FOR STAGE 2 OBJECTIVE SELECTION")
    logger.info("=" * 80)

    avg_solutions = summary_df["solution_count"].mean()
    max_solutions = summary_df["solution_count"].max()
    levels_with_multiple = len(summary_df[summary_df["solution_count"] > 1])

    logger.info("\nAlternative Solutions Statistics:")
    logger.info(f"  Average alternatives per spending level: {avg_solutions:.1f}")
    logger.info(f"  Maximum alternatives at any level: {max_solutions:.0f}")
    logger.info(
        f"  Spending levels with multiple solutions: {levels_with_multiple}/{len(spending_levels)}"
    )

    if levels_with_multiple > 0:
        logger.info("\nMetric Variation Across Alternative Solutions:")
        logger.info(f"  Revenue range (avg): ${summary_df['revenue_range'].mean():,.0f}B")
        logger.info(f"  Jobs range (avg): {summary_df['jobs_range'].mean():,.0f}")

        # Determine which metric shows most variation
        revenue_variation = summary_df["revenue_range"].mean()
        jobs_variation = summary_df["jobs_range"].mean()

        # Check Stage 2 effectiveness
        stage2_counts = summary_df["stage2_solution_count"]
        still_multiple = len(stage2_counts[stage2_counts > 1])

        logger.info("\nStage 2 Effectiveness:")
        logger.info(
            f"  Spending levels with multiple solutions after Stage 2: {still_multiple}/{len(spending_levels)}"
        )
        if still_multiple > 0:
            logger.info(f"  Maximum alternatives after Stage 2: {stage2_counts.max():.0f}")
            logger.info("  → Stage 2 reduces but doesn't eliminate all alternatives")
            logger.info("  → Stage 3 objective (revenue) needed")

            # Check Stage 3 effectiveness
            if "stage3_solution_count" in summary_df.columns:
                stage3_counts = summary_df["stage3_solution_count"]
                still_multiple_stage3 = len(stage3_counts[stage3_counts > 1])

                logger.info("\nStage 3 Effectiveness:")
                logger.info(
                    f"  Spending levels with multiple solutions after Stage 3: {still_multiple_stage3}/{len(spending_levels)}"
                )
                if still_multiple_stage3 > 0:
                    logger.info(f"  Maximum alternatives after Stage 3: {stage3_counts.max():.0f}")
                    logger.info("  → Stage 3 reduces but doesn't eliminate all alternatives")
                    logger.info("  → STAGE 4 NEEDED!")

                    # Analyze what varies in Stage 3 alternatives
                    if "stage3_capital_range" in summary_df.columns:
                        avg_capital_var = summary_df["stage3_capital_range"].mean()
                        avg_wage_var = summary_df["stage3_wage_range"].mean()
                        avg_equity_var = summary_df["stage3_equity_ratio_range"].mean()

                        logger.info("\n  Variation in Stage 3 Alternatives:")
                        logger.info(f"    Capital variation (avg): {avg_capital_var:.6f}")
                        logger.info(f"    Wage variation (avg): {avg_wage_var:.6f}")
                        logger.info(f"    Equity ratio variation (avg): {avg_equity_var:.4f}")

                        logger.info("\n  RECOMMENDED STAGE 4 OBJECTIVE:")
                        if avg_equity_var > 0.1:
                            logger.info(
                                "    → Maximize equity ratio (P20/P99) - shows highest variation"
                            )
                        elif avg_capital_var > 0.001:
                            logger.info("    → Maximize capital stock - shows meaningful variation")
                        elif avg_wage_var > 0.0001:
                            logger.info("    → Maximize wage rate - shows some variation")
                        else:
                            logger.info(
                                "    → Solutions are effectively identical - Stage 4 not meaningful"
                            )
                else:
                    logger.info("  → Stage 3 produces UNIQUE solutions for all spending levels")
                    logger.info("  → THREE STAGES ARE SUFFICIENT - No Stage 4 needed")
        else:
            logger.info("  → Stage 2 produces UNIQUE solutions for all spending levels")
            logger.info("  → No need for Stage 3")

        logger.info("\nRECOMMENDATION:")
        if jobs_variation > revenue_variation * 0.001:  # Jobs vary significantly
            logger.info(
                "  ✅ Jobs show significant variation across alternatives - maximizing jobs in Stage 2 is meaningful"
            )
        elif revenue_variation > 100:  # Revenue varies by >$100B  # noqa: PLR2004
            logger.info(
                "  Revenue shows significant variation - maximizing revenue in Stage 2 is meaningful"
            )
        else:
            logger.info(
                "  Limited variation in both jobs and revenue - Stage 2 objective has minimal impact"
            )
    else:
        logger.info(
            "\nAll spending levels have unique optimal solutions - Stage 2 objective is not needed"
        )


if __name__ == "__main__":
    main()
